"""사람이 확정한 장면만 집계한다. 촬영하지 않은 기록은 0으로 채우지 않는다."""

import math


KINDS = {"shot", "oreb", "dreb", "assist", "steal", "block", "turnover", "foul"}
STAT_COLUMNS = [
    "PTS", "FGM", "FGA", "FG%", "3PM", "3PA", "3P%", "FTM", "FTA", "FT%",
    "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF",
]
# 장면 종류를 기록표의 열 이름으로 바꾼다. 슛은 점수 구분에 따라 따로 계산한다.
COUNT_MAP = {
    "oreb": "OREB",
    "dreb": "DREB",
    "assist": "AST",
    "steal": "STL",
    "block": "BLK",
    "turnover": "TOV",
    "foul": "PF",
}


def validate_event(event, players, duration):
    """입력한 기록을 검사한다. 문제가 있으면 화면에 보여줄 오류를 발생시킨다."""
    if event.get("kind") not in KINDS:
        raise ValueError("기록 종류를 선택해주세요.")

    try:
        event_time = float(event.get("time", 0))
    except (ValueError, TypeError):
        raise ValueError("기록 시간이 올바르지 않습니다.")

    if not math.isfinite(event_time) or not 0 <= event_time <= duration:
        raise ValueError("기록 시간이 영상 범위를 벗어났습니다.")
    event["time"] = round(event_time, 3)

    status = event.get("status", "pending")
    if status not in {"pending", "confirmed", "rejected"}:
        raise ValueError("기록 상태가 올바르지 않습니다.")

    player_id = event.get("player_id") or None
    registered_ids = []
    for player in players:
        registered_ids.append(player["id"])

    if player_id and player_id not in registered_ids:
        raise ValueError("등록된 선수를 선택해주세요.")
    event["player_id"] = player_id

    if status == "confirmed" and not player_id:
        raise ValueError("선수를 지정해야 기록을 확정할 수 있습니다.")

    if event["kind"] == "shot":
        if event.get("result", "unknown") not in {"made", "missed", "unknown"}:
            raise ValueError("슛 결과가 올바르지 않습니다.")
        if event.get("value") not in {None, 1, 2, 3}:
            raise ValueError("자유투·2점·3점 중 선택해주세요.")

        result_is_known = event.get("result") in {"made", "missed"}
        value_is_known = event.get("value") in {1, 2, 3}
        if status == "confirmed" and (not result_is_known or not value_is_known):
            raise ValueError("슛 결과와 자유투·2점·3점을 확인해야 확정할 수 있습니다.")

    return event


def has_confirmed_stat(events, player_id, column):
    """확정된 부가 기록 중 해당 선수와 기록표 열에 맞는 장면이 있는지 찾는다."""
    for event in events:
        if event.get("player_id") != player_id:
            continue
        if COUNT_MAP.get(event["kind"]) == column:
            return True
    return False


def boxscore(session):
    """확정 기록을 처음부터 합산하므로 기존 장면을 수정해도 중복 집계되지 않는다."""
    mode = session.get("mode", "offense")
    events = session.get("events", [])
    confirmed_events = []
    pending_count = 0
    for event in events:
        if event.get("status") == "confirmed":
            confirmed_events.append(event)
        if event.get("status") == "pending":
            pending_count += 1

    rows = []
    for player in session.get("players", []):
        counts = {}
        for column in STAT_COLUMNS:
            counts[column] = 0

        for event in confirmed_events:
            if event.get("player_id") != player["id"]:
                continue

            kind = event["kind"]
            if kind == "shot":
                shot_value = event["value"]
                shot_was_made = event["result"] == "made"

                if shot_value == 1:
                    counts["FTA"] += 1
                    if shot_was_made:
                        counts["FTM"] += 1
                else:
                    counts["FGA"] += 1
                    if shot_was_made:
                        counts["FGM"] += 1
                    if shot_value == 3:
                        counts["3PA"] += 1
                        if shot_was_made:
                            counts["3PM"] += 1

                if shot_was_made:
                    counts["PTS"] += shot_value
            elif kind in COUNT_MAP:
                column = COUNT_MAP[kind]
                counts[column] += 1

        # 성공 수 ÷ 시도 수 × 100. 시도가 없다면 성공률은 계산할 수 없다.
        rate_columns = [
            ("FG%", "FGM", "FGA"),
            ("3P%", "3PM", "3PA"),
            ("FT%", "FTM", "FTA"),
        ]
        for rate_column, made_column, attempt_column in rate_columns:
            attempts = counts[attempt_column]
            if attempts:
                counts[rate_column] = round(100 * counts[made_column] / attempts, 1)
            else:
                counts[rate_column] = None

        # 하이라이트에서는 실패 장면이 빠질 수 있어 시도 수와 성공률을 숨긴다.
        # 부가 기록은 사용자가 직접 확정한 항목만 보여준다.
        if mode == "highlights":
            incomplete_columns = [
                "FGA", "FG%", "3PA", "3P%", "FTA", "FT%", "OREB", "DREB",
                "AST", "STL", "BLK", "TOV", "PF",
            ]
            for column in incomplete_columns:
                if column not in COUNT_MAP.values():
                    counts[column] = None
                elif not has_confirmed_stat(confirmed_events, player["id"], column):
                    counts[column] = None
        elif mode == "offense":
            for column in ["DREB", "STL", "BLK", "PF"]:
                if not has_confirmed_stat(confirmed_events, player["id"], column):
                    counts[column] = None

        row = {
            "player_id": player["id"],
            "name": player["name"],
            "team": player.get("team", ""),
            "number": player.get("number", ""),
        }
        row.update(counts)
        rows.append(row)

    return {
        "rows": rows,
        "columns": STAT_COLUMNS,
        "confirmed": len(confirmed_events),
        "pending": pending_count,
        "scope": "영상에서 확인·확정한 장면만 집계합니다. —는 미집계이며, 0과 다릅니다.",
    }
