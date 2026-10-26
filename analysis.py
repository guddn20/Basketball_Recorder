"""영상에서 사람·공을 찾아 추적 구간과 슛 후보를 만든다.

자동 분석은 후보만 제안한다. 선수·성공 여부·점수는 사람이 확인해야 한다.
box 좌표는 [왼쪽, 위쪽, 오른쪽, 아래쪽]이며 영상 크기에 대한 0~1 비율이다.
"""

from collections import deque
from pathlib import Path
import math
import threading
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# 모델은 한 번만 읽어 재사용하고, 동시에 여러 분석이 모델을 사용하지 않게 한다.
MODEL_LOCK = threading.Lock()
_MODEL = None


def get_model():
    """처음 사용할 때만 모델을 읽는다. 모델 파일은 setup.ps1로 미리 준비한다."""
    global _MODEL
    if _MODEL is None:
        import os
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        os.environ.setdefault("YOLO_OFFLINE", "true")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).parent / "data" / "yolo_config"))

        import torch
        from ultralytics import YOLO

        cpu_threads = min(4, max(1, (os.cpu_count() or 4) // 2))
        torch.set_num_threads(cpu_threads)
        model_path = Path(__file__).parent / "models" / "yolo11n.pt"
        if not model_path.exists():
            raise RuntimeError("models/yolo11n.pt가 없습니다. setup.ps1을 실행해 모델을 준비해주세요.")
        _MODEL = YOLO(str(model_path))
    return _MODEL


def frame_at(path, seconds):
    """지정한 시간(초)의 영상 이미지 한 장을 읽는다."""
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0, seconds) * 1000)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise ValueError("해당 시간의 프레임을 읽을 수 없습니다. 영상 끝보다 조금 앞으로 이동해주세요.")
    return frame


def crop_box(frame, box):
    """비율 좌표를 실제 이미지의 픽셀 좌표로 바꿔 사각형 부분을 잘라낸다."""
    height, width = frame.shape[:2]
    left, top, right, bottom = box
    top_pixel = max(0, int(top * height))
    bottom_pixel = min(height, int(bottom * height))
    left_pixel = max(0, int(left * width))
    right_pixel = min(width, int(right * width))
    # NumPy 이미지는 [세로 범위, 가로 범위] 순서로 자른다.
    return frame[top_pixel:bottom_pixel, left_pixel:right_pixel]


def appearance(frame, box):
    """상의·하의·신발 영역의 색 분포를 숫자 목록으로 만든다."""
    cropped_person = crop_box(frame, box)
    if cropped_person.size == 0:
        return None

    height, width = cropped_person.shape[:2]
    features = []
    # 주변 선수와 바닥이 덜 섞이도록 전신의 중앙 부분만 비교한다.
    body_regions = [(0.20, 0.53), (0.55, 0.78), (0.82, 1.0)]
    for top, bottom in body_regions:
        top_pixel = int(top * height)
        bottom_pixel = max(top_pixel + 1, int(bottom * height))
        left_pixel = int(0.15 * width)
        right_pixel = max(left_pixel + 1, int(0.85 * width))
        region = cropped_person[top_pixel:bottom_pixel, left_pixel:right_pixel]

        # HSV는 색상·채도·밝기를 구분하는 색 표현 방식이다.
        hsv_image = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv_image], [0, 1, 2], None,
            [12, 4, 4], [0, 180, 0, 256, 0, 256],
        )
        histogram = cv2.normalize(
            histogram, None, alpha=1, norm_type=cv2.NORM_L1,
        ).flatten()
        features.append(histogram.tolist())
    return features


def similarity(a, b):
    """두 전신의 색 분포를 비교한다. 값이 클수록 비슷하며 확률은 아니다."""
    if a is None or b is None:
        return 0.0

    region_scores = []
    for first_region, second_region in zip(a, b):
        first_histogram = np.asarray(first_region, np.float32)
        second_histogram = np.asarray(second_region, np.float32)
        distance = cv2.compareHist(
            first_histogram, second_histogram, cv2.HISTCMP_BHATTACHARYYA,
        )
        region_scores.append(1 - distance)

    # 기존 비교 비중을 유지한다. 상의 38%, 하의 20%, 신발 42%이다.
    weighted_score = 0
    for weight, score in zip([0.38, 0.20, 0.42], region_scores):
        weighted_score += weight * score
    return float(weighted_score)


def suggest_player(feature, players):
    """가장 비슷한 선수를 제안하되, 두 후보가 비슷하면 이름을 확정하지 않는다."""
    player_scores = []
    for player in players:
        sample_scores = []
        for sample in player.get("samples", []):
            sample_scores.append(similarity(feature, sample["feature"]))

        if sample_scores:
            best_sample_score = max(sample_scores)
        else:
            best_sample_score = 0
        player_scores.append((best_sample_score, player["id"]))

    player_scores.sort(reverse=True)
    if not player_scores:
        return None, 0

    best_score, player_id = player_scores[0]
    if len(player_scores) > 1:
        score_margin = best_score - player_scores[1][0]
    else:
        score_margin = best_score

    # 이 수치는 기존 후보 선택 기준이며 측정된 인식 정확도가 아니다.
    if best_score >= 0.57 and score_margin >= 0.06:
        return player_id, round(best_score, 3)
    return None, round(best_score, 3)


def iou(a, b):
    """두 사각형이 겹치는 면적을 전체 합집합 면적으로 나눈다."""
    overlap_width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = overlap_width * overlap_height
    first_area = (a[2] - a[0]) * (a[3] - a[1])
    second_area = (b[2] - b[0]) * (b[3] - b[1])
    union = first_area + second_area - intersection
    return intersection / max(1e-8, union)


class Tracker:
    """가까운 위치와 비슷한 외형을 이용해 짧은 구간의 사람 박스를 연결한다."""

    def __init__(self):
        self.next_id = 1
        self.active = {}

    def reset(self):
        # 장면이 바뀌어도 번호를 되돌리지 않아 이전 구간과 ID가 겹치지 않는다.
        self.active.clear()

    def update(self, people, t):
        current_time = t
        recent_tracks = {}
        for track_id, track in self.active.items():
            if current_time - track["time"] <= 1.0:
                recent_tracks[track_id] = track
        self.active = recent_tracks

        track_ids = list(self.active)
        # 행은 이전 추적 대상, 열은 현재 사람이다. 비용이 작을수록 연결하기 좋다.
        costs = np.full((len(track_ids), len(people)), 10.0, np.float32)
        for track_index, track_id in enumerate(track_ids):
            previous_person = self.active[track_id]
            for person_index, person in enumerate(people):
                previous_box = previous_person["box"]
                current_box = person["box"]
                horizontal_distance = (
                    previous_box[0] + previous_box[2] - current_box[0] - current_box[2]
                ) / 2
                vertical_distance = (
                    previous_box[1] + previous_box[3] - current_box[1] - current_box[3]
                ) / 2
                distance = math.hypot(horizontal_distance, vertical_distance)
                appearance_score = similarity(previous_person["feature"], person["feature"])
                if distance <= 0.16 and appearance_score >= 0.18:
                    overlap_cost = 0.52 * (1 - iou(previous_box, current_box))
                    appearance_cost = 0.33 * (1 - appearance_score)
                    distance_cost = 0.15 * min(1, distance / 0.12)
                    costs[track_index, person_index] = overlap_cost + appearance_cost + distance_cost

        matched_people = set()
        if costs.size:
            # SciPy가 전체 비용이 작은 일대일 연결을 찾는다. 알고리즘 구현은 라이브러리에 맡긴다.
            track_indexes, person_indexes = linear_sum_assignment(costs)
            for track_index, person_index in zip(track_indexes, person_indexes):
                if costs[track_index, person_index] < 0.82:
                    people[person_index]["track_id"] = track_ids[track_index]
                    matched_people.add(int(person_index))

        for person_index, person in enumerate(people):
            if person_index not in matched_people:
                person["track_id"] = str(self.next_id)
                self.next_id += 1
            saved_person = dict(person)
            saved_person["time"] = current_time
            self.active[person["track_id"]] = saved_person
        return people


class HoopLocator:
    """고정 촬영은 등록 좌표를 쓰고, 이동 촬영은 골대 주변 이미지와 비교한다."""

    def __init__(self, config, path, moving):
        self.config = config
        self.moving = moving
        self.template = None
        self.rim_offset = None
        self.last = config["box"]
        if moving:
            frame = frame_at(path, config["time"])
            height, width = frame.shape[:2]
            left, top, right, bottom = config["box"]
            rim_width = max(0.015, right - left)
            context_box = [
                max(0, left - rim_width),
                max(0, top - 2 * rim_width),
                min(1, right + rim_width),
                min(1, bottom + rim_width),
            ]
            resized_frame = cv2.resize(frame, (960, round(height * 960 / width)))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            self.template = crop_box(gray_frame, context_box)
            # 비교용 이미지의 왼쪽 위를 기준으로 림이 어디에 있는지 저장한다.
            self.rim_offset = [
                left - context_box[0], top - context_box[1],
                right - context_box[0], bottom - context_box[1],
            ]

    def locate(self, frame):
        if not self.moving:
            return self.config["box"], 1.0

        height, width = frame.shape[:2]
        resized_frame = cv2.resize(frame, (960, round(height * 960 / width)))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        if self.template is None or self.template.size == 0:
            return None, 0
        for template_size, frame_size in zip(self.template.shape, gray_frame.shape):
            if template_size > frame_size:
                return None, 0

        matches = cv2.matchTemplate(gray_frame, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(matches)
        if score < 0.58:
            return None, float(score)

        gray_height, gray_width = gray_frame.shape
        origin_x = location[0] / gray_width
        origin_y = location[1] / gray_height
        rim_box = [
            origin_x + self.rim_offset[0], origin_y + self.rim_offset[1],
            origin_x + self.rim_offset[2], origin_y + self.rim_offset[3],
        ]
        return rim_box, float(score)


def orange_balls(frame, rim, previous_gray):
    """골대 주변에서 크기·색·움직임이 공과 비슷한 영역을 찾는다."""
    height, width = frame.shape[:2]
    left, top, right, bottom = rim
    rim_width = max(right - left, 0.012)
    search_box = [
        max(0, left - 2 * rim_width), max(0, top - 5 * rim_width),
        min(1, right + 2 * rim_width), min(1, bottom + 3 * rim_width),
    ]
    cropped_frame = crop_box(frame, search_box)
    if cropped_frame.size == 0:
        return []

    hsv_image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
    # mask는 조건에 맞는 픽셀만 흰색으로 남긴 이미지이다.
    mask = cv2.inRange(hsv_image, np.array([2, 95, 75]), np.array([27, 255, 255]))
    if previous_gray is not None:
        current_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        previous_region = crop_box(previous_gray, search_box)
        motion = cv2.absdiff(current_gray, previous_region)
        mask = cv2.bitwise_and(mask, cv2.inRange(motion, 18, 255))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, ball_width, ball_height = cv2.boundingRect(contour)
        minimum_area = max(3, (rim_width * width * 0.09) ** 2)
        maximum_area = (rim_width * width * 0.95) ** 2
        aspect_ratio = ball_width / max(1, ball_height)
        if not minimum_area <= area <= maximum_area:
            continue
        if not 0.35 < aspect_ratio < 2.8:
            continue

        center_x = search_box[0] + (x + ball_width / 2) / width
        center_y = search_box[1] + (y + ball_height / 2) / height
        # 주황색인 림 자체를 공으로 잘못 찾지 않도록 제외한다.
        inside_rim_x = left - 0.003 <= center_x <= right + 0.003
        inside_rim_y = top - 0.004 <= center_y <= bottom + 0.004
        if inside_rim_x and inside_rim_y:
            continue

        balls.append({
            "x": round(center_x, 5), "y": round(center_y, 5),
            "conf": 0.25, "source": "color",
        })
    return balls[:8]


def release_balls(frame, people, previous_gray):
    """선수의 손·머리 근처 주황색 움직임을 찾아 슈터 추정에만 사용한다."""
    height, width = frame.shape[:2]
    balls = []
    for person in people:
        left, top, right, bottom = person["box"]
        person_height = bottom - top
        person_width = right - left
        search_box = [
            max(0, left - person_width * 0.2),
            max(0, top - person_height * 0.25),
            min(1, right + person_width * 0.2),
            min(1, top + person_height * 0.42),
        ]
        cropped_frame = crop_box(frame, search_box)
        if cropped_frame.size == 0:
            continue

        hsv_image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, np.array([3, 145, 90]), np.array([26, 255, 255]))
        if previous_gray is not None:
            current_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            previous_region = crop_box(previous_gray, search_box)
            motion = cv2.absdiff(current_gray, previous_region)
            mask = cv2.bitwise_and(mask, cv2.inRange(motion, 22, 255))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, ball_width, ball_height = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * math.pi * area / max(1, perimeter * perimeter)
            minimum_area = max(6, (person_height * height * 0.017) ** 2)
            maximum_area = (person_height * height * 0.13) ** 2
            aspect_ratio = ball_width / max(1, ball_height)
            if not minimum_area <= area <= maximum_area:
                continue
            if not 0.45 < aspect_ratio < 2 or circularity < 0.35:
                continue

            center_x = search_box[0] + (x + ball_width / 2) / width
            center_y = search_box[1] + (y + ball_height / 2) / height
            balls.append({
                "x": round(center_x, 5), "y": round(center_y, 5),
                "conf": 0.2, "source": "release_color",
                "track_hint": person["track_id"],
            })
    return balls


def suggest_shooter(history, current_time, center_x, center_y, rim_width):
    """직전 영상에서 공과 어깨가 가까웠던 선수를 제안한다. 득점은 확정하지 않는다."""
    shooter = None
    best_rank = 1e9
    for previous_frame in history:
        elapsed = current_time - previous_frame["t"]
        if not 0.25 <= elapsed <= 2.6:
            continue
        for ball in previous_frame["balls"]:
            distance_to_rim = math.hypot(ball["x"] - center_x, ball["y"] - center_y)
            if ball["source"] not in {"model", "release_color"} or distance_to_rim < rim_width * 2:
                continue
            for person in previous_frame["people"]:
                track_hint = ball.get("track_hint")
                if track_hint and track_hint != person["track_id"]:
                    continue
                box = person["box"]
                shoulder_x = (box[0] + box[2]) / 2
                shoulder_y = box[1] + (box[3] - box[1]) * 0.2
                distance = math.hypot(ball["x"] - shoulder_x, ball["y"] - shoulder_y)
                rank = distance + elapsed * 0.003
                if distance < 0.09 and rank < best_rank:
                    best_rank = rank
                    shooter = person
    return shooter


def run_analysis(path, session, options, progress, cancelled):
    """지정 구간을 분석한다. progress는 진행 표시, cancelled는 중지 여부 확인 함수이다."""
    start = options["start"]
    end = options["end"]
    sample_fps = options.get("fps", 5)
    capture = cv2.VideoCapture(str(path))
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # 예: 30fps 영상을 5fps로 분석하면 6장마다 한 장을 사용한다.
    frame_stride = max(1, round(video_fps / sample_fps))
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    tracker = Tracker()
    frames = []
    candidates = []
    track_summaries = {}
    history = deque()
    near_rim_history = {}
    last_candidate_time = {}
    previous_small = None
    previous_gray = None
    scene_number = 0
    scanned_frames = 0
    frame_index = round(start * video_fps)
    hoop_locators = []
    for hoop in session.get("hoops", []):
        if hoop["end"] >= start and hoop["start"] <= end:
            moving_camera = session.get("camera") == "moving"
            hoop_locators.append(HoopLocator(hoop, path, moving_camera))

    progress(0, "모델 준비 중 · 첫 실행에는 잠시 걸릴 수 있습니다.")
    with MODEL_LOCK:
        model = get_model()
        try:
            while True:
                current_time = frame_index / video_fps
                if current_time > end or cancelled():
                    break
                success, frame = capture.read()
                if not success:
                    break
                frame_index += 1
                if (frame_index - 1 - round(start * video_fps)) % frame_stride:
                    continue

                # 1. 화면이 크게 바뀌면 이전 장면의 추적과 공 이동 기록을 끊는다.
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small_frame = cv2.resize(gray_frame, (64, 36))
                scene_changed = False
                if previous_small is not None:
                    frame_difference = np.mean(cv2.absdiff(small_frame, previous_small))
                    scene_changed = frame_difference > 53
                if scene_changed:
                    scene_number += 1
                    tracker.reset()
                    history.clear()
                    near_rim_history.clear()
                    previous_gray = None
                previous_small = small_frame

                # 2. COCO 분류 번호 0(사람), 32(스포츠 공)만 감지한다.
                result = model.predict(
                    frame, classes=[0, 32], imgsz=960,
                    conf=0.16, verbose=False, device="cpu",
                )[0]
                people = []
                balls = []
                for detection in result.boxes:
                    class_id = int(detection.cls.item())
                    confidence = float(detection.conf.item())
                    pixel_box = detection.xyxy[0].tolist()
                    box = [
                        round(pixel_box[0] / width, 5), round(pixel_box[1] / height, 5),
                        round(pixel_box[2] / width, 5), round(pixel_box[3] / height, 5),
                    ]
                    if class_id == 0 and confidence >= 0.27 and box[3] - box[1] > 0.035:
                        feature = appearance(frame, box)
                        player_id, identity_score = suggest_player(feature, session.get("players", []))
                        people.append({
                            "box": box, "conf": round(confidence, 3), "feature": feature,
                            "suggested_player_id": player_id, "identity_score": identity_score,
                        })
                    elif class_id == 32:
                        balls.append({
                            "x": round((box[0] + box[2]) / 2, 5),
                            "y": round((box[1] + box[3]) / 2, 5),
                            "conf": round(confidence, 3), "source": "model",
                        })

                people = tracker.update(people, current_time)
                balls.extend(release_balls(frame, people, previous_gray))

                # 3. 등록한 골대가 적용되는 시간인지 확인하고 주변 공을 살핀다.
                located_hoops = []
                for locator in hoop_locators:
                    hoop = locator.config
                    if not hoop["start"] <= current_time <= hoop["end"]:
                        continue
                    rim, location_score = locator.locate(frame)
                    if rim is None:
                        continue
                    located_hoops.append({
                        "id": hoop["id"], "box": rim, "score": round(location_score, 3),
                    })
                    balls.extend(orange_balls(frame, rim, previous_gray))
                    left, top, right, bottom = rim
                    rim_width = max(0.012, right - left)
                    center_x = (left + right) / 2
                    center_y = (top + bottom) / 2
                    nearby_balls = []
                    for ball in balls:
                        close_x = abs(ball["x"] - center_x) < rim_width * 1.55
                        close_y = center_y - rim_width * 2.8 < ball["y"] < center_y + rim_width * 2.0
                        if close_x and close_y:
                            nearby_balls.append(ball)

                    recent_balls = near_rim_history.setdefault(hoop["id"], deque())
                    for ball in nearby_balls:
                        recent_balls.append((current_time, ball))
                    while recent_balls and recent_balls[0][0] < current_time - 1.3:
                        recent_balls.popleft()

                    ball_was_above = False
                    for _, ball in recent_balls:
                        if ball["y"] < center_y - rim_width * 0.22:
                            ball_was_above = True
                            break
                    ball_is_crossing = False
                    model_found_ball = False
                    for ball in nearby_balls:
                        if abs(ball["y"] - center_y) < rim_width * 1.1:
                            ball_is_crossing = True
                        if ball["source"] == "model":
                            model_found_ball = True

                    # 같은 골대에서 2.7초 안에 후보를 반복 생성하지 않는다.
                    time_since_candidate = current_time - last_candidate_time.get(hoop["id"], -100)
                    color_suggests_shot = ball_was_above and ball_is_crossing and len(recent_balls) >= 2
                    if nearby_balls and time_since_candidate > 2.7 and (color_suggests_shot or model_found_ball):
                        shooter = suggest_shooter(history, current_time, center_x, center_y, rim_width)
                        player_id = None
                        track_id = None
                        if shooter:
                            player_id = shooter.get("suggested_player_id")
                            track_id = shooter.get("track_id")
                        confidence = max(ball["conf"] for ball in nearby_balls)
                        candidates.append({
                            "time": round(current_time, 3), "kind": "shot",
                            "result": "unknown", "value": None, "status": "pending",
                            "player_id": player_id, "track_id": track_id,
                            "source": "auto",
                            "note": "공이 골대 주변을 지난 후보입니다. 슛 여부·선수·성공/실패·점수를 영상에서 확인해주세요.",
                            "confidence": round(confidence, 3), "hoop_id": hoop["id"],
                        })
                        last_candidate_time[hoop["id"]] = current_time
                        recent_balls.clear()

                # 4. 각 추적 구간을 요약하고 화면에 필요한 분석 결과를 모은다.
                stored_people = []
                for person in people:
                    track_id = person["track_id"]
                    if track_id not in track_summaries:
                        track_summaries[track_id] = {
                            "id": track_id, "start": round(current_time, 3),
                            "end": round(current_time, 3), "count": 0, "scene": scene_number,
                            "sample_time": round(current_time, 3), "sample_box": person["box"],
                            "votes": {},
                        }
                    summary = track_summaries[track_id]
                    summary["end"] = round(current_time, 3)
                    summary["count"] += 1
                    player_id = person["suggested_player_id"]
                    if player_id:
                        previous_votes = summary["votes"].get(player_id, 0)
                        summary["votes"][player_id] = previous_votes + person["identity_score"]

                    stored_person = {}
                    for key, value in person.items():
                        if key != "feature":
                            stored_person[key] = value
                    stored_people.append(stored_person)

                row = {
                    "t": round(current_time, 3), "people": stored_people,
                    "balls": balls, "hoops": located_hoops, "scene": scene_number,
                }
                frames.append(row)
                history.append(row)
                while history and history[0]["t"] < current_time - 3:
                    history.popleft()
                previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                scanned_frames += 1
                if scanned_frames % 5 == 0:
                    percent = min(99, 100 * (current_time - start) / max(0.01, end - start))
                    message = (
                        f"{current_time - start:.0f}/{end - start:.0f}초 분석 · "
                        f"선수 구간 {len(track_summaries)}개 · 슛 후보 {len(candidates)}개"
                    )
                    progress(percent, message)
        finally:
            capture.release()

    # 구간 전체에서 누적한 외형 비교 점수로 대표 이름 후보를 정한다.
    for summary in track_summaries.values():
        votes = summary.pop("votes")
        ranked_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        summary["suggested_player_id"] = None
        summary["match_score"] = 0
        if ranked_votes:
            player_id, total_score = ranked_votes[0]
            average_score = total_score / summary["count"]
            if average_score > 0.35:
                summary["suggested_player_id"] = player_id
            summary["match_score"] = round(average_score, 3)

    if frames:
        processed_end = min(end, (frame_index - 1) / video_fps)
    else:
        processed_end = start
    return {
        "frames": frames, "tracks": list(track_summaries.values()),
        "candidates": candidates, "start": start, "end": processed_end,
        "sample_fps": sample_fps, "cancelled": cancelled(), "created": time.time(),
    }
