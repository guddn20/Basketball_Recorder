import copy
import unittest

from stats import boxscore, validate_event


class RecordContractTests(unittest.TestCase):
    def setUp(self):
        self.session={"mode":"full","players":[{"id":"p1","name":"주황색 신발","team":"흰색"}],"events":[]}

    def event(self, **changes):
        return {"id":"e1","time":4,"kind":"shot","player_id":"p1","result":"made","value":3,"status":"confirmed",**changes}

    def test_pending_candidates_never_change_score(self):
        self.session["events"]=[self.event(status="pending"),self.event(status="rejected")]
        stats=boxscore(self.session)
        self.assertEqual(stats["rows"][0]["PTS"],0)
        self.assertEqual(stats["rows"][0]["FGA"],0)
        self.assertEqual(stats["pending"],1)

    def test_field_goals_threes_and_free_throws_are_distinct(self):
        self.session["events"]=[self.event(),self.event(value=2,result="missed"),self.event(value=1),self.event(value=1,result="missed")]
        row=boxscore(self.session)["rows"][0]
        self.assertEqual([row[k] for k in ["PTS","FGM","FGA","3PM","3PA","FTM","FTA"]],[4,1,2,1,1,1,2])
        self.assertEqual(row["FG%"],50)
        self.assertEqual(row["FT%"],50)

    def test_highlights_never_report_perfect_shooting_rate(self):
        self.session["mode"]="highlights"
        self.session["events"]=[self.event()]
        row=boxscore(self.session)["rows"][0]
        self.assertEqual(row["FGM"],1)
        for key in ["FGA","FG%","3PA","3P%","FTA","FT%","DREB","STL"]:
            self.assertIsNone(row[key],key)

    def test_offense_does_not_turn_missing_defense_into_zeros(self):
        self.session["mode"]="offense"
        row=boxscore(self.session)["rows"][0]
        for key in ["DREB","STL","BLK","PF"]:
            self.assertIsNone(row[key],key)
        self.assertEqual(row["OREB"],0)

    def test_manual_defensive_record_is_preserved_even_in_partial_video(self):
        self.session["mode"]="offense"
        self.session["events"]=[self.event(kind="steal")]
        self.assertEqual(boxscore(self.session)["rows"][0]["STL"],1)

    def test_cannot_confirm_unidentified_player_or_unknown_score(self):
        for changes in [{"player_id":None},{"player_id":"unknown"},{"value":None},{"result":"unknown"},{"time":float("nan")},{"time":51}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_event(self.event(**changes),self.session["players"],50)

    def test_pending_unknown_shot_can_be_saved_without_fabricated_value(self):
        event=self.event(status="pending",value=None,result="unknown",player_id=None)
        self.assertEqual(validate_event(event,self.session["players"],50),event)

    def test_editing_confirmed_event_recalculates_without_double_count(self):
        self.session["events"]=[self.event()]
        self.assertEqual(boxscore(self.session)["rows"][0]["PTS"],3)
        self.session["events"][0]["value"]=2
        self.assertEqual(boxscore(self.session)["rows"][0]["PTS"],2)
        self.session["events"][0]["status"]="rejected"
        self.assertEqual(boxscore(self.session)["rows"][0]["PTS"],0)


if __name__=="__main__":
    unittest.main()
