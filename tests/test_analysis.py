"""모델 다운로드 없이 영상 분석의 입력부터 후보 생성까지 확인한다.

가짜 모델은 정해진 위치만 반환한다. 실제 농구 영상의 인식 정확도는 측정하지 않는다.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import analysis


class MemoryVideo:
    """디스크의 영상 대신 같은 크기의 이미지 목록을 순서대로 읽는다."""

    def __init__(self, frames):
        self.frames = frames
        self.index = 0
        self.released = False

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FPS:
            return 10
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.frames[0].shape[1]
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frames[0].shape[0]
        return 0

    def set(self, property_id, value):
        if property_id == cv2.CAP_PROP_POS_MSEC:
            self.index = round(value / 1000 * 10)
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index].copy()
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


class ScriptedModel:
    """처음에는 선수 어깨에, 이후에는 골대 주변에 공이 있다고 응답한다."""

    def __init__(self):
        self.calls = 0

    def detection(self, class_id, box):
        # YOLO 응답에서 사용하는 item(), tolist()만 NumPy 값으로 제공한다.
        return SimpleNamespace(
            cls=np.array(class_id),
            conf=np.array(0.9),
            xyxy=np.array([box]),
        )

    def predict(self, frame, **options):
        if self.calls < 2:
            ball_box = [38, 40, 42, 44]
        else:
            ball_box = [148, 18, 152, 22]
        self.calls += 1
        detections = [
            self.detection(0, [20, 30, 60, 90]),
            self.detection(32, ball_box),
        ]
        return [SimpleNamespace(boxes=detections)]


class AnalysisFlowTests(unittest.TestCase):
    def setUp(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[30:90, 20:60] = 200
        self.frames = [frame.copy() for _ in range(12)]
        player_box = [0.1, 0.3, 0.3, 0.9]
        feature = analysis.appearance(frame, player_box)
        self.session = {
            "camera": "fixed",
            "players": [{"id": "player-1", "samples": [{"feature": feature}]}],
            "hoops": [{
                "id": "hoop-1", "box": [0.7, 0.2, 0.8, 0.25],
                "time": 0, "start": 0, "end": 1,
            }],
        }

    def run_video(self, cancelled):
        video = MemoryVideo(self.frames)
        progress_messages = []

        def record_progress(value, message):
            progress_messages.append((value, message))

        with patch.object(analysis.cv2, "VideoCapture", return_value=video):
            with patch.object(analysis, "get_model", return_value=ScriptedModel()):
                result = analysis.run_analysis(
                    "memory-video", self.session,
                    {"start": 0, "end": 1, "fps": 5},
                    record_progress, cancelled,
                )
        return result, video, progress_messages

    def test_full_flow_samples_tracks_and_proposes_reviewable_shot(self):
        result, video, messages = self.run_video(lambda: False)

        # 원래 10fps인 영상에서 한 장씩 건너뛰어 5fps의 분석 결과를 만든다.
        self.assertEqual([frame["t"] for frame in result["frames"]], [0, 0.2, 0.4, 0.6, 0.8, 1])
        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(result["tracks"][0]["suggested_player_id"], "player-1")
        self.assertEqual(len(result["candidates"]), 1)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["status"], "pending")
        self.assertEqual(candidate["result"], "unknown")
        self.assertIsNone(candidate["value"])
        self.assertEqual(candidate["player_id"], "player-1")
        for frame in result["frames"]:
            self.assertNotIn("feature", frame["people"][0])
        self.assertTrue(video.released)
        self.assertTrue(messages)

    def test_cancel_after_processing_keeps_only_processed_frames(self):
        # 반복 중 다섯 번째 확인에서 중지한다. 분석한 이미지는 0초와 0.2초이다.
        cancel_checks = 0

        def cancelled():
            nonlocal cancel_checks
            cancel_checks += 1
            return cancel_checks >= 5

        result, video, _ = self.run_video(cancelled)
        self.assertTrue(result["cancelled"])
        self.assertEqual([frame["t"] for frame in result["frames"]], [0, 0.2])
        self.assertEqual(result["end"], 0.3)
        self.assertTrue(video.released)

    def test_cancel_before_first_frame_produces_no_replacement_results(self):
        result, video, _ = self.run_video(lambda: True)
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["frames"], [])
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["end"], 0)
        self.assertTrue(video.released)


if __name__ == "__main__":
    unittest.main()
