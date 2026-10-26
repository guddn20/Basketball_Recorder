import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
import server


class LocalWorkflowTests(unittest.TestCase):
    def setUp(self):
        test_root=(server.ROOT/"test-results").resolve()
        test_root.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=test_root)
        assert Path(self.temp.name).resolve().is_relative_to(test_root)
        self.patch=patch.object(server,"DATA",Path(self.temp.name))
        self.patch.start()
        server.JOBS.clear()
        self.client=TestClient(server.app)
        path=Path(self.temp.name)/"fixture.mp4"
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),10,(320,180))
        for _ in range(20):
            frame=np.zeros((180,320,3),np.uint8)
            cv2.rectangle(frame,(80,30),(110,160),(230,230,230),-1)
            cv2.rectangle(frame,(80,145),(110,160),(0,100,255),-1)
            writer.write(frame)
        writer.release()
        with path.open("rb") as stream:
            response=self.client.post("/api/upload",files={"file":("test.mp4",stream,"video/mp4")})
        self.assertEqual(response.status_code,200,response.text)
        self.session=response.json()
        self.base="/api/sessions/"+self.session["id"]

    def tearDown(self):
        self.client.close()
        self.patch.stop()
        self.temp.cleanup()

    def add_player(self,name="테스트 선수"):
        response=self.client.post(self.base+"/players",json={"time":.5,"box":[.24,.15,.36,.91],"name":name,"team":"흰색"})
        self.assertEqual(response.status_code,200,response.text)
        return response.json()["players"][0]["id"]

    def test_upload_frame_and_seekable_video(self):
        response=self.client.get(self.base+"/video",headers={"Range":"bytes=0-99"})
        self.assertEqual(response.status_code,206)
        self.assertEqual(len(response.content),100)
        suffix=self.client.get(self.base+"/video",headers={"Range":"bytes=-10"})
        self.assertEqual(len(suffix.content),10)
        self.assertEqual(self.client.get(self.base+"/video",headers={"Range":"bytes=999999999-"}).status_code,416)
        frame=self.client.get(self.base+"/frame?t=0.5&crop=.24,.15,.36,.91")
        self.assertEqual(frame.status_code,200)
        self.assertEqual(frame.headers["content-type"],"image/jpeg")

    def test_player_reference_features_remain_private_to_local_store(self):
        self.add_player()
        public=self.client.get(self.base).json()
        self.assertNotIn("feature",public["players"][0]["samples"][0])
        stored=server.read_session(self.session["id"])
        self.assertIn("feature",stored["players"][0]["samples"][0])

    def test_review_confirm_edit_reject_and_csv(self):
        pid=self.add_player("=2+2")
        response=self.client.post(self.base+"/events",json={"time":.8,"kind":"shot","status":"pending"})
        s=response.json();event=s["events"][0]
        self.assertEqual(s["stats"]["rows"][0]["PTS"],0)
        invalid=self.client.patch(self.base+"/events/"+event["id"],json={"status":"confirmed"})
        self.assertEqual(invalid.status_code,400)
        url=self.base+"/events/"+event["id"]
        s=self.client.patch(url,json={"player_id":pid,"result":"made","value":3,"status":"confirmed"}).json()
        self.assertEqual(s["stats"]["rows"][0]["PTS"],3)
        csv=self.client.get(self.base+"/export/boxscore").content.decode("utf-8-sig")
        self.assertIn("'=2+2",csv)
        self.assertIn("미집계",csv)
        s=self.client.patch(url,json={"value":2}).json()
        self.assertEqual(s["stats"]["rows"][0]["PTS"],2)
        self.assertEqual(len(s["events"]),1)
        s=self.client.patch(url,json={"status":"rejected"}).json()
        self.assertEqual(s["stats"]["rows"][0]["PTS"],0)
        backup=self.client.get(self.base+"/export/json").json()
        self.assertEqual(backup["events"][0]["status"],"rejected")

    def test_hoop_intervals_cannot_overlap_or_exclude_reference_time(self):
        payload={"time":.5,"start":0,"end":1,"box":[.4,.2,.46,.23]}
        self.assertEqual(self.client.post(self.base+"/hoops",json=payload).status_code,200)
        self.assertEqual(self.client.post(self.base+"/hoops",json=payload).status_code,400)
        self.assertEqual(self.client.post(self.base+"/hoops",json={**payload,"start":1,"end":2,"time":.5}).status_code,400)

    def test_cross_origin_writes_are_rejected(self):
        response=self.client.patch(self.base,json={"mode":"full"},headers={"Origin":"https://example.org"})
        self.assertEqual(response.status_code,403)

    def test_reanalysis_keeps_manual_confirmed_and_rejected_events(self):
        pid=self.add_player()
        sid=self.session["id"]
        s=server.read_session(sid)
        base={"time":.5,"kind":"shot","player_id":pid,"result":"made","value":2,"note":""}
        s["events"]=[{**base,"id":"manual","status":"pending","source":"manual"},
                     {**base,"id":"old_auto","status":"pending","source":"auto"},
                     {**base,"id":"confirmed","status":"confirmed","source":"auto"},
                     {**base,"id":"rejected","status":"rejected","source":"auto"}]
        server.save_session(s)
        result={"frames":[{"t":0,"people":[]}],"tracks":[],"candidates":[{**base,"status":"pending","source":"auto"}],"start":0,"end":1,"sample_fps":5,"cancelled":False}
        server.JOBS[sid]={"state":"running"}
        with patch.object(server,"run_analysis",return_value=result):
            server.worker(sid,{"start":0,"end":1})
        s=server.read_session(sid)
        self.assertEqual({e["id"] for e in s["events"]},{"manual","confirmed","rejected"})
        self.assertEqual(server.JOBS[sid]["state"],"done")

    def test_cancelled_analysis_records_only_processed_range(self):
        sid=self.session["id"]
        result={"frames":[{"t":0,"people":[]}],"tracks":[],"candidates":[],"start":0,"end":.5,"sample_fps":5,"cancelled":True}
        server.JOBS[sid]={"state":"running","cancel_requested":True}
        with patch.object(server,"run_analysis",return_value=result): server.worker(sid,{"start":0,"end":2})
        s=server.read_session(sid)
        self.assertEqual(s["analyses"][0]["end"],.5)
        self.assertEqual(server.JOBS[sid]["state"],"cancelled")

    def test_cancelling_before_first_frame_preserves_existing_analysis(self):
        sid=self.session["id"]
        analysis_path=server.sid_path(sid)/"analysis.json"
        analysis_path.write_text('{"frames":[{"t":0.5}],"tracks":[]}',"utf-8")
        original=analysis_path.read_bytes()
        result={"frames":[],"tracks":[],"candidates":[],"start":0,"end":0,"sample_fps":5,"cancelled":True}
        server.JOBS[sid]={"state":"running","cancel_requested":True}
        with patch.object(server,"run_analysis",return_value=result): server.worker(sid,{"start":0,"end":2})
        self.assertEqual(analysis_path.read_bytes(),original)
        self.assertEqual(server.JOBS[sid]["state"],"cancelled")


if __name__=="__main__":
    unittest.main()
