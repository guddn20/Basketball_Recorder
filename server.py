"""브라우저의 요청을 받아 영상·선수·기록을 저장하고 분석 작업을 실행한다.

sid는 영상 ID, pid는 선수 ID, hid는 골대 ID, eid는 기록 ID, tid는 추적 구간 ID이다.
이 이름들은 API 주소의 자리 이름이며 기존 호출과 호환되도록 유지한다.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from analysis import appearance, crop_box, frame_at, run_analysis
from stats import boxscore, validate_event


ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("COURTSIDE_DATA", ROOT / "data"))
DATA.mkdir(parents=True, exist_ok=True)
# 저장 중 다른 요청이 같은 기록을 덮어쓰지 않게 한다.
# RLock은 같은 작업 안에서 read_session/save_session을 다시 호출해도 사용할 수 있다.
LOCK = threading.RLock()
# 영상 분석은 한 번에 하나만 실행하고 진행 상태는 JOBS에 보관한다.
EXECUTOR = ThreadPoolExecutor(max_workers=1)
JOBS = {}
MAX_UPLOAD = 8 * 1024**3

app = FastAPI(title="코트 기록실", docs_url=None, redoc_url=None)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "testserver"],
)


@app.middleware("http")
async def local_origin(request: Request, call_next):
    """각 요청 앞뒤에서 실행한다. 다른 사이트에서 보내는 변경 요청은 거절한다."""
    origin = request.headers.get("origin")
    if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
        from urllib.parse import urlparse
        if urlparse(origin).hostname not in {"127.0.0.1", "localhost"}:
            return Response("Local requests only", status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def sid_path(sid):
    """영상 ID를 검사한 다음 저장 폴더 경로를 반환한다."""
    if len(sid) != 32:
        raise HTTPException(404, "영상을 찾을 수 없습니다.")
    for character in sid:
        if character not in "0123456789abcdef":
            raise HTTPException(404, "영상을 찾을 수 없습니다.")
    return DATA / sid


def read_session(sid):
    """영상 하나에 연결된 선수·골대·기록 정보를 읽는다."""
    with LOCK:
        try:
            session_path = sid_path(sid) / "session.json"
            return json.loads(session_path.read_text("utf-8"))
        except FileNotFoundError:
            raise HTTPException(404, "영상을 찾을 수 없습니다.")


def save_session(session):
    """임시 파일을 다 쓴 뒤 교체해서 저장 중인 JSON을 읽는 일을 줄인다."""
    with LOCK:
        session["updated"] = time.time()
        session_path = sid_path(session["id"]) / "session.json"
        temporary_path = session_path.with_suffix(".tmp")
        text = json.dumps(session, ensure_ascii=False, indent=2)
        temporary_path.write_text(text, "utf-8")
        os.replace(temporary_path, session_path)


def public_session(session):
    """화면용 복사본을 만든다. 파일명과 내부 색상 특징은 일반 응답에서 뺀다."""
    public = copy.deepcopy(session)
    public.pop("file", None)
    for player in public.get("players", []):
        for sample in player.get("samples", []):
            sample.pop("feature", None)
    public["stats"] = boxscore(session)
    public["job"] = JOBS.get(session["id"], {"state": "idle", "progress": 0})
    return public


def video_path(session):
    return sid_path(session["id"]) / session["file"]


def normalized_box(value):
    """드래그한 사각형이 영상 안에 있고 충분한 크기인지 검사한다."""
    try:
        box = []
        for coordinate in value:
            box.append(float(coordinate))
    except (ValueError, TypeError):
        raise HTTPException(400, "영상에서 사각형 영역을 지정해주세요.")

    invalid_box_message = "사각형 영역이 너무 작거나 영상 범위를 벗어났습니다."
    if len(box) != 4:
        raise HTTPException(400, invalid_box_message)
    for coordinate in box:
        if not math.isfinite(coordinate) or not 0 <= coordinate <= 1:
            raise HTTPException(400, invalid_box_message)
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    if box_width < 0.003 or box_height < 0.003:
        raise HTTPException(400, invalid_box_message)

    rounded_box = []
    for coordinate in box:
        rounded_box.append(round(coordinate, 6))
    return rounded_box


def finite_time(value, duration):
    """숫자로 바꿀 수 있고 영상 길이 안에 있는 시간인지 검사한다."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise HTTPException(400, "시간을 숫자로 입력해주세요.")
    if not math.isfinite(seconds) or not 0 <= seconds <= duration:
        raise HTTPException(400, "시간이 영상 범위를 벗어났습니다.")
    return seconds


def find_by_id(items, item_id):
    """선수·기록 등의 목록에서 ID가 같은 첫 항목을 찾는다."""
    for item in items:
        if item["id"] == item_id:
            return item
    return None


def import_video(source, name=None, mode="offense"):
    """로컬 파일을 복사해 등록한다. 원본 영상은 수정하지 않는다."""
    sid = uuid.uuid4().hex
    directory = sid_path(sid)
    directory.mkdir()
    suffix = Path(source).suffix.lower()
    target = directory / ("video" + suffix)
    shutil.copyfile(source, target)
    return finish_import(sid, target, name or Path(source).name, mode)


def finish_import(sid, target, name, mode="offense"):
    """복사·업로드한 영상의 정보를 읽고 초기 기록 파일을 만든다."""
    capture = cv2.VideoCapture(str(target))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    success, preview = capture.read()
    capture.release()
    if not success or not fps or not frame_count or width < 32 or height < 32:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "영상을 읽을 수 없습니다. MP4(H.264) 파일을 권장합니다.")

    duration = frame_count / fps
    if not math.isfinite(duration) or duration <= 0:
        raise HTTPException(400, "영상의 길이를 읽을 수 없습니다.")

    preview_height = round(height * 640 / width)
    preview = cv2.resize(preview, (640, preview_height))
    cv2.imwrite(str(sid_path(sid) / "poster.jpg"), preview)
    session = {
        "id": sid, "name": name[:200], "file": target.name,
        "created": time.time(), "updated": time.time(),
        "duration": round(duration, 3), "width": width, "height": height, "fps": fps,
        "mode": mode, "camera": "fixed", "players": [], "hoops": [], "events": [],
        "track_assignments": {}, "analyses": [],
    }
    save_session(session)
    return session


# @app.get/post/patch/delete는 어떤 주소와 요청 방식에서 함수를 실행할지 정한다.
@app.get("/api/health")
def health():
    return {
        "app": "courtside-local", "version": "0.1.0",
        "pid": os.getpid(), "root": str(ROOT),
        "model_ready": (ROOT / "models/yolo11n.pt").exists(),
    }


@app.get("/api/sessions")
def sessions():
    result = []
    for path in DATA.glob("*/session.json"):
        try:
            session = json.loads(path.read_text("utf-8"))
            summary = {}
            for key in ["id", "name", "duration", "created", "updated"]:
                summary[key] = session[key]
            result.append(summary)
        except (ValueError, KeyError, OSError):
            continue
    # lambda는 여기서 정렬에 사용할 '수정 시간'을 꺼내는 짧은 함수이다.
    return sorted(result, key=lambda session: session["updated"], reverse=True)


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}:
        raise HTTPException(400, "MP4, MOV, M4V, AVI, MKV, WEBM 영상을 선택해주세요.")

    sid = uuid.uuid4().hex
    directory = sid_path(sid)
    directory.mkdir()
    target = directory / ("video" + suffix)
    uploaded_size = 0
    try:
        # 큰 영상을 한꺼번에 메모리에 올리지 않고 8MB씩 나누어 저장한다.
        with target.open("wb") as output:
            while True:
                chunk = file.file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                uploaded_size += len(chunk)
                if uploaded_size > MAX_UPLOAD:
                    raise HTTPException(413, "영상은 최대 8GB까지 추가할 수 있습니다.")
                output.write(chunk)
        session = finish_import(sid, target, file.filename or "영상")
        return public_session(session)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        file.file.close()


@app.get("/api/sessions/{sid}")
def session_get(sid: str):
    return public_session(read_session(sid))


@app.patch("/api/sessions/{sid}")
def session_update(sid: str, payload: dict):
    with LOCK:
        session = read_session(sid)
        if "name" in payload:
            session["name"] = str(payload["name"]).strip()[:200] or session["name"]
        if "mode" in payload:
            if payload["mode"] not in {"offense", "highlights", "full"}:
                raise HTTPException(400, "촬영 범위를 선택해주세요.")
            session["mode"] = payload["mode"]
        if "camera" in payload:
            if payload["camera"] not in {"fixed", "moving"}:
                raise HTTPException(400, "촬영 방식을 선택해주세요.")
            session["camera"] = payload["camera"]
        save_session(session)
        return public_session(session)


@app.get("/api/sessions/{sid}/video")
def media(sid: str, request: Request):
    """브라우저가 요청한 파일 구간만 전송해 영상의 중간 시간으로 이동할 수 있게 한다."""
    path = video_path(read_session(sid))
    file_size = path.stat().st_size
    mime_types = {
        ".mp4": "video/mp4", ".m4v": "video/mp4",
        ".mov": "video/quicktime", ".webm": "video/webm",
    }
    mime_type = mime_types.get(path.suffix, "application/octet-stream")
    range_header = request.headers.get("range")
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"}
    start_byte = 0
    end_byte = file_size - 1
    status_code = 200
    if range_header:
        try:
            unit, span = range_header.split("=", 1)
            left, right = span.split("-", 1)
            if unit != "bytes" or "," in span:
                raise ValueError()
            if left:
                start_byte = int(left)
                if right:
                    end_byte = min(int(right), file_size - 1)
                else:
                    end_byte = file_size - 1
            else:
                # bytes=-10은 파일 끝의 10바이트를 달라는 뜻이다.
                requested_length = int(right)
                if requested_length <= 0:
                    raise ValueError()
                start_byte = max(0, file_size - requested_length)
            if start_byte < 0 or start_byte > end_byte or start_byte >= file_size:
                raise ValueError()
        except ValueError:
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{file_size}"},
            )
        status_code = 206
        headers["Content-Range"] = f"bytes {start_byte}-{end_byte}/{file_size}"
    headers["Content-Length"] = str(end_byte - start_byte + 1)

    def chunks():
        """yield로 파일 조각을 하나씩 보내 메모리 사용을 줄인다."""
        with path.open("rb") as stream:
            stream.seek(start_byte)
            remaining = end_byte - start_byte + 1
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        chunks(), status_code=status_code, media_type=mime_type, headers=headers,
    )


@app.get("/api/sessions/{sid}/frame")
def still(sid: str, t: float = 0, crop: str = "", width: int = 960):
    session = read_session(sid)
    last_frame_time = max(0, session["duration"] - 1 / session["fps"])
    frame_time = finite_time(t, last_frame_time)
    try:
        frame = frame_at(video_path(session), frame_time)
        if crop:
            frame = crop_box(frame, normalized_box(crop.split(",")))
        height, original_width = frame.shape[:2]
        width = min(1920, max(80, width))
        if original_width > width:
            resized_height = max(1, round(height * width / original_width))
            frame = cv2.resize(frame, (width, resized_height))
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("프레임 변환에 실패했습니다.")
        return Response(encoded.tobytes(), media_type="image/jpeg")
    except ValueError as error:
        raise HTTPException(400, str(error))


@app.post("/api/sessions/{sid}/players")
def player_add(sid: str, payload: dict):
    with LOCK:
        session = read_session(sid)
        box = normalized_box(payload.get("box"))
        last_frame_time = max(0, session["duration"] - 1 / session["fps"])
        sample_time = finite_time(payload.get("time", 0), last_frame_time)
        name = str(payload.get("name", "")).strip()[:60]
        existing_player = find_by_id(session["players"], payload.get("player_id"))
        if not existing_player and not name:
            raise HTTPException(400, "선수 이름을 입력해주세요.")
        try:
            frame = frame_at(video_path(session), sample_time)
            feature = appearance(frame, box)
        except ValueError as error:
            raise HTTPException(400, str(error))

        if existing_player:
            player = existing_player
        else:
            player = {
                "id": uuid.uuid4().hex, "name": name,
                "team": str(payload.get("team", "")).strip()[:40],
                "number": str(payload.get("number", "")).strip()[:10],
                "note": str(payload.get("note", "")).strip()[:300],
                "samples": [],
            }
            session["players"].append(player)
        if len(player["samples"]) >= 12:
            raise HTTPException(400, "선수별 참고 장면은 최대 12개입니다.")
        player["samples"].append({
            "time": round(sample_time, 3), "box": box, "feature": feature,
        })
        save_session(session)
        return public_session(session)


@app.patch("/api/sessions/{sid}/players/{pid}")
def player_update(sid: str, pid: str, payload: dict):
    with LOCK:
        session = read_session(sid)
        player = find_by_id(session["players"], pid)
        if player is None:
            raise HTTPException(404, "선수를 찾을 수 없습니다.")
        for key, limit in [("name", 60), ("team", 40), ("number", 10), ("note", 300)]:
            if key in payload:
                value = str(payload[key]).strip()[:limit]
                if key == "name" and not value:
                    raise HTTPException(400, "이름을 입력해주세요.")
                player[key] = value
        save_session(session)
        return public_session(session)


@app.post("/api/sessions/{sid}/hoops")
def hoop_add(sid: str, payload: dict):
    with LOCK:
        session = read_session(sid)
        box = normalized_box(payload.get("box"))
        start = finite_time(payload.get("start", 0), session["duration"])
        end = finite_time(payload.get("end", session["duration"]), session["duration"])
        last_frame_time = max(0, session["duration"] - 1 / session["fps"])
        reference_time = finite_time(payload.get("time", start), last_frame_time)
        if start >= end or not start <= reference_time <= end:
            raise HTTPException(400, "골대 기준 장면이 적용 구간 안에 있어야 합니다.")

        # 같은 시간대에 공격 골대가 겹치면 후보가 중복 생성될 수 있다.
        for hoop in session["hoops"]:
            overlap_start = max(start, hoop["start"])
            overlap_end = min(end, hoop["end"])
            if overlap_start < overlap_end - 0.01:
                raise HTTPException(400, "골대 구간이 겹칩니다. 기존 구간을 삭제하거나 시간을 나눠주세요.")
        session["hoops"].append({
            "id": uuid.uuid4().hex, "name": str(payload.get("name", "공격 골대"))[:60],
            "box": box, "time": reference_time, "start": start, "end": end,
        })
        session["hoops"].sort(key=lambda hoop: hoop["start"])
        save_session(session)
        return public_session(session)


@app.delete("/api/sessions/{sid}/hoops/{hid}")
def hoop_delete(sid: str, hid: str):
    with LOCK:
        session = read_session(sid)
        remaining_hoops = []
        for hoop in session["hoops"]:
            if hoop["id"] != hid:
                remaining_hoops.append(hoop)
        session["hoops"] = remaining_hoops
        save_session(session)
        return public_session(session)


def update_event_fields(event, payload):
    """화면에서 수정할 수 있는 항목만 복사한다. ID와 기록 방식은 유지한다."""
    allowed_fields = {"time", "kind", "player_id", "result", "value", "status", "note"}
    for key, value in payload.items():
        if key in allowed_fields:
            event[key] = value


@app.post("/api/sessions/{sid}/events")
def event_add(sid: str, payload: dict):
    with LOCK:
        session = read_session(sid)
        event = {
            "id": uuid.uuid4().hex, "source": "manual", "status": "pending",
            "result": "unknown", "value": None, "note": "",
        }
        update_event_fields(event, payload)
        try:
            validate_event(event, session["players"], session["duration"])
        except ValueError as error:
            raise HTTPException(400, str(error))
        session["events"].append(event)
        save_session(session)
        return public_session(session)


@app.patch("/api/sessions/{sid}/events/{eid}")
def event_update(sid: str, eid: str, payload: dict):
    with LOCK:
        session = read_session(sid)
        event = find_by_id(session["events"], eid)
        if event is None:
            raise HTTPException(404, "기록을 찾을 수 없습니다.")
        update_event_fields(event, payload)
        try:
            validate_event(event, session["players"], session["duration"])
        except ValueError as error:
            raise HTTPException(400, str(error))
        save_session(session)
        return public_session(session)


@app.get("/api/sessions/{sid}/analysis")
def analysis_get(sid: str):
    read_session(sid)
    path = sid_path(sid) / "analysis.json"
    if not path.exists():
        return {"frames": [], "tracks": [], "ranges": []}
    return json.loads(path.read_text("utf-8"))


@app.post("/api/sessions/{sid}/tracks/{tid}")
def track_assign(sid: str, tid: str, payload: dict):
    with LOCK:
        session = read_session(sid)
        player_id = payload.get("player_id") or None
        if player_id and find_by_id(session["players"], player_id) is None:
            raise HTTPException(400, "등록된 선수를 선택해주세요.")
        analysis_result = analysis_get(sid)
        if find_by_id(analysis_result["tracks"], tid) is None:
            raise HTTPException(404, "추적 구간을 찾을 수 없습니다.")

        session["track_assignments"][tid] = player_id
        # 추적 구간 수정은 미확정 후보에만 반영한다. 확정한 과거 기록은 그대로 둔다.
        for event in session["events"]:
            same_track = event.get("track_id") == tid
            if same_track and event["status"] == "pending":
                event["player_id"] = player_id
        save_session(session)
        return public_session(session)


def prefix_track_ids(result, analysis_id):
    """분석마다 다른 접두사를 붙여 추적 번호가 겹치지 않게 한다."""
    for frame in result["frames"]:
        for person in frame["people"]:
            person["track_id"] = analysis_id + "_" + person["track_id"]
    for track in result["tracks"]:
        track["id"] = analysis_id + "_" + track["id"]
    for candidate in result["candidates"]:
        if candidate.get("track_id"):
            candidate["track_id"] = analysis_id + "_" + candidate["track_id"]


def merge_analysis(previous, result):
    """새로 처리한 시간 구간만 교체하고 다른 시간의 분석은 보존한다."""
    start = result["start"]
    end = result["end"]
    remaining_frames = []
    for frame in previous["frames"]:
        if not start <= frame["t"] <= end:
            remaining_frames.append(frame)
    previous["frames"] = remaining_frames + result["frames"]
    previous["frames"].sort(key=lambda frame: frame["t"])

    used_track_ids = set()
    for frame in previous["frames"]:
        for person in frame["people"]:
            used_track_ids.add(person["track_id"])
    remaining_tracks = []
    for track in previous["tracks"]:
        if track["id"] in used_track_ids:
            remaining_tracks.append(track)
    previous["tracks"] = remaining_tracks + result["tracks"]
    previous.setdefault("ranges", []).append({
        "start": start, "end": end,
        "fps": result["sample_fps"], "cancelled": result["cancelled"],
    })


def replace_pending_candidates(session, result):
    """자동 미확정 후보만 교체하고 직접 입력·확정·제외 기록은 유지한다."""
    remaining_events = []
    for event in session["events"]:
        is_automatic = event.get("source") == "auto"
        is_pending = event["status"] == "pending"
        in_processed_range = result["start"] <= event["time"] <= result["end"]
        if not (is_automatic and is_pending and in_processed_range):
            remaining_events.append(event)
    session["events"] = remaining_events

    for candidate in result["candidates"]:
        already_reviewed = False
        for event in session["events"]:
            close_in_time = abs(event["time"] - candidate["time"]) < 1.4
            is_reviewed_shot = event["kind"] == "shot" and event["status"] in {"confirmed", "rejected"}
            if close_in_time and is_reviewed_shot:
                already_reviewed = True
                break
        if already_reviewed:
            continue
        candidate["id"] = uuid.uuid4().hex
        session["events"].append(candidate)


def worker(sid, options):
    """별도 스레드에서 분석하고 완료 또는 중지한 범위를 저장한다."""
    job = JOBS[sid]

    def progress(value, message):
        job.update(progress=round(value, 1), message=message)

    def is_cancelled():
        return job.get("cancel_requested", False)

    try:
        snapshot = read_session(sid)
        result = run_analysis(
            video_path(snapshot), snapshot, options, progress, is_cancelled,
        )
        if not result["frames"] and result["cancelled"]:
            job.update(
                state="cancelled", progress=0,
                message="분석을 중지했습니다. 새 결과가 없어 기존 기록을 유지했습니다.",
            )
            return
        if not result["frames"]:
            raise ValueError("선택한 구간에서 프레임을 읽지 못했습니다. 시작 시간을 앞쪽으로 이동해주세요.")

        analysis_id = uuid.uuid4().hex[:8]
        prefix_track_ids(result, analysis_id)
        with LOCK:
            session = read_session(sid)
            previous_analysis = analysis_get(sid)
            merge_analysis(previous_analysis, result)

            target = sid_path(sid) / "analysis.json"
            temporary_path = target.with_suffix(".tmp")
            text = json.dumps(previous_analysis, ensure_ascii=False)
            temporary_path.write_text(text, "utf-8")
            os.replace(temporary_path, target)

            replace_pending_candidates(session, result)
            session["analyses"].append({
                "id": analysis_id, "start": result["start"], "end": result["end"],
                "frames": len(result["frames"]), "candidates": len(result["candidates"]),
                "cancelled": result["cancelled"], "created": time.time(),
            })
            save_session(session)

        if result["cancelled"]:
            state = "cancelled"
        else:
            state = "done"
        message = f"{len(result['frames'])}개 프레임 · 슛 후보 {len(result['candidates'])}개. 후보를 확인해주세요."
        job.update(state=state, progress=100, message=message)
    except Exception as error:
        import traceback
        error_path = sid_path(sid) / "last_error.log"
        error_path.write_text(traceback.format_exc(), "utf-8")
        job.update(
            state="error",
            message=f"분석을 완료하지 못했습니다: {str(error)[:300]}. 수동 기록은 계속 사용할 수 있습니다.",
        )


@app.post("/api/sessions/{sid}/analyze")
def analyze(sid: str, payload: dict):
    with LOCK:
        session = read_session(sid)
        for job in JOBS.values():
            if job["state"] == "running":
                raise HTTPException(409, "다른 분석이 진행 중입니다. 완료하거나 중지한 뒤 실행해주세요.")
        start = finite_time(payload.get("start", 0), session["duration"])
        default_end = min(session["duration"], start + 60)
        end = finite_time(payload.get("end", default_end), session["duration"])
        if end - start < 0.5:
            raise HTTPException(400, "분석 구간은 최소 0.5초 이상으로 지정해주세요.")
        sample_fps = payload.get("fps", 5)
        if sample_fps not in {3, 5, 10}:
            raise HTTPException(400, "분석 속도를 선택해주세요.")
        JOBS[sid] = {
            "state": "running", "progress": 0,
            "message": "분석을 시작합니다.", "cancel_requested": False,
        }
        options = {"start": start, "end": end, "fps": sample_fps}
        EXECUTOR.submit(worker, sid, options)
        return JOBS[sid]


@app.post("/api/sessions/{sid}/cancel")
def cancel(sid: str):
    read_session(sid)
    if JOBS.get(sid, {}).get("state") == "running":
        JOBS[sid]["cancel_requested"] = True
        JOBS[sid]["message"] = "현재 프레임 처리 후 중지합니다. 처리한 구간은 저장됩니다."
    return JOBS.get(sid, {"state": "idle"})


@app.get("/api/sessions/{sid}/export/{kind}")
def export(sid: str, kind: str):
    session = read_session(sid)
    if kind == "json":
        backup = copy.deepcopy(session)
        backup.pop("file", None)
        backup["stats"] = boxscore(session)
        content = json.dumps(backup, ensure_ascii=False, indent=2).encode("utf-8")
        return Response(
            content, media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="courtside-records.json"'},
        )

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    if kind == "boxscore":
        stats = boxscore(session)
        columns = stats["columns"]
        writer.writerow(["선수", "팀", "등번호", "집계범위", "영상종류"] + columns)
        for row in stats["rows"]:
            values = [row["name"], row["team"], row["number"], "영상 내 확정 장면", session["mode"]]
            for column in columns:
                value = row.get(column)
                if value is None:
                    value = "미집계"
                values.append(value)
            writer.writerow(values)
    elif kind == "events":
        writer.writerow(["영상초", "선수", "종류", "결과", "점수구분", "상태", "기록방식", "메모"])
        player_names = {}
        for player in session["players"]:
            player_names[player["id"]] = player["name"]
        for event in sorted(session["events"], key=lambda event: event["time"]):
            writer.writerow([
                event["time"], player_names.get(event.get("player_id"), "미지정"),
                event["kind"], event.get("result", ""), event.get("value", ""),
                event["status"], event.get("source", ""), event.get("note", ""),
            ])
    else:
        raise HTTPException(404, "내보내기 형식이 올바르지 않습니다.")

    # 사용자 입력이 Excel 수식으로 실행되지 않도록 특수문자 앞에 따옴표를 붙인다.
    csv_rows = list(csv.reader(io.StringIO(output.getvalue())))
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    for row in csv_rows:
        safe_row = []
        for value in row:
            if value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            safe_row.append(value)
        writer.writerow(safe_row)
    return Response(
        output.getvalue().encode("utf-8-sig"), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="courtside-{kind}.csv"'},
    )


# API 주소를 먼저 등록한 뒤 나머지 주소에서 HTML/CSS/JavaScript 파일을 제공한다.
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, host="127.0.0.1",
        port=int(os.environ.get("COURTSIDE_PORT", "8765")), log_level="warning",
    )
