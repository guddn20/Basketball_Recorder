import unittest
from unittest.mock import patch

import cv2
import numpy as np

from analysis import appearance, similarity, suggest_player, Tracker, HoopLocator, orange_balls, release_balls


class TrackingTests(unittest.TestCase):
    def person_image(self,shoe):
        frame=np.full((200,100,3),40,np.uint8)
        frame[40:106,15:85]=235
        frame[110:156,15:85]=20
        frame[164:200,15:85]=shoe
        return frame

    def test_shoe_color_helps_separate_similar_white_shirts(self):
        orange=appearance(self.person_image((0,100,255)),[0,0,1,1])
        blue=appearance(self.person_image((255,30,0)),[0,0,1,1])
        self.assertGreater(similarity(orange,orange),similarity(orange,blue))
        players=[{"id":"orange","samples":[{"feature":orange}]},{"id":"blue","samples":[{"feature":blue}]}]
        self.assertEqual(suggest_player(orange,players)[0],"orange")
        self.assertEqual(suggest_player(blue,players)[0],"blue")

    def test_ambiguous_same_outfit_does_not_force_an_identity(self):
        feature=appearance(self.person_image((0,100,255)),[0,0,1,1])
        players=[{"id":pid,"samples":[{"feature":feature}]} for pid in ["p1","p2"]]
        self.assertIsNone(suggest_player(feature,players)[0])

    def test_short_tracks_persist_and_scene_reset_uses_new_ids(self):
        feature=appearance(self.person_image((0,100,255)),[0,0,1,1])
        tracker=Tracker()
        first=tracker.update([{"box":[.1,.2,.2,.7],"feature":feature}],0)[0]["track_id"]
        second=tracker.update([{"box":[.12,.2,.22,.7],"feature":feature}],.2)[0]["track_id"]
        self.assertEqual(first,second)
        tracker.reset()
        third=tracker.update([{"box":[.12,.2,.22,.7],"feature":feature}],.4)[0]["track_id"]
        self.assertNotEqual(first,third)

    def test_moving_camera_rim_template_follows_translation(self):
        rng=np.random.default_rng(42)
        frame=rng.integers(0,255,(540,960,3),dtype=np.uint8)
        box=[.45,.3,.5,.33]
        config={"time":0,"box":box}
        with patch("analysis.frame_at",return_value=frame): locator=HoopLocator(config,"unused",True)
        shifted=cv2.warpAffine(frame,np.float32([[1,0,30],[0,1,15]]),(960,540))
        found,score=locator.locate(shifted)
        self.assertGreater(score,.95)
        self.assertAlmostEqual(found[0],box[0]+30/960,places=2)
        self.assertAlmostEqual(found[1],box[1]+15/540,places=2)

    def test_static_orange_rim_is_not_a_moving_ball(self):
        frame=np.zeros((540,960,3),np.uint8)
        cv2.rectangle(frame,(430,160),(490,175),(0,100,255),-1)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.assertEqual(orange_balls(frame,[430/960,160/540,490/960,175/540],gray),[])

    def test_moving_orange_ball_near_head_proposes_a_track_not_a_score(self):
        frame=self.person_image((0,100,255))
        cv2.circle(frame,(50,20),6,(0,100,255),-1)
        people=[{"box":[0,0,1,1],"track_id":"p4"}]
        hints=release_balls(frame,people,np.zeros((200,100),np.uint8))
        self.assertTrue(hints)
        self.assertEqual(hints[0]["track_hint"],"p4")
        self.assertNotIn("value",hints[0])


if __name__=="__main__": unittest.main()
