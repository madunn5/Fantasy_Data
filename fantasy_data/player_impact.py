"""Player impact: who actually swung games, and what roster moves paid off.

Three different questions, three different baselines:

- league_edge(): every started player measured against the league's median
  starter at his position that week. This is the superstar table. A player
  only "swings" a win when the team would have lost with an ordinary starter
  in his slot, so having a bad backup earns no credit here.
- costly_benchings(): the manager table. Losses where starting a benched
  player over the weakest legal starter would have won the game. Grouped per
  game, since one loss with three viable bench options is still one loss.
- roster_moves(): every player who changed fantasy teams mid-season and what
  he produced after arriving. When the Yahoo transaction log has been
  collected (PlayerTransaction), trades are labelled from it exactly;
  otherwise reciprocal moves between the same two teams in the same window
  are grouped as a best-guess trade.

Swap legality follows the lineup: QB/K/DEF only swap within their position;
WR/RB/TE can displace any started WR/RB/TE because of the flex slot.
"""
from collections import defaultdict
from statistics import median

from .models import (PlayerPerformance, PlayerTransaction, TeamPerformance,
                     TeamOwnerMapping)
from .predictions import week_number

FLEX_POSITIONS = {'WR', 'RB', 'TE'}

# Lineup shape used to strip IR pollution from stored data (see _clean_starters).
_MAX_FLEX_STARTERS = 6  # 2 WR + 2 RB + 1 TE + 1 flex


def _game_results(year):
    """{(team, week_num): {'result': 'W'/'L', 'margin': float}}"""
    out = {}
    for r in TeamPerformance.objects.filter(year=year).values(
            "team_name", "week", "result", "margin"):
        out[(r["team_name"], week_number(r["week"]))] = {
            "result": (r["result"] or "").upper(),
            "margin": float(r["margin"] or 0),
        }
    return out


def _clean_starters(starters):
    """Strip probable IR pollution from a list of started players.

    Data collected before the IR fix counts injured-reserve players as
    starters. The lineup only has 6 WR/RB/TE spots and one each of QB/K/DEF,
    so anything beyond that is an IR body: drop the lowest scorers among the
    flex-eligible overflow and keep the best QB/K/DEF of any duplicates.
    """
    flexers = sorted((p for p in starters if p["pos"] in FLEX_POSITIONS),
                     key=lambda p: p["pts"])
    while len(flexers) > _MAX_FLEX_STARTERS:
        flexers.pop(0)
    out = list(flexers)
    for pos in ("QB", "K", "DEF"):
        ps = sorted((p for p in starters if p["pos"] == pos),
                    key=lambda p: -p["pts"])
        out += ps[:1]
    return out


def _rosters(year):
    """{(team, week_num): {'start': [player], 'bench': [player]}} with IR cleaned.

    Each player is ``{'id', 'name', 'pos', 'pts'}``. Keyed by player id, not
    name — two NFL defenses can share a display name like "Los Angeles".
    """
    grouped = defaultdict(lambda: {"start": [], "bench": []})
    rows = PlayerPerformance.objects.filter(year=year).values(
        "fantasy_team", "week", "points_scored", "was_started",
        "player_id", "player__name", "player__position")
    for r in rows:
        p = {"id": r["player_id"], "name": r["player__name"],
             "pos": r["player__position"], "pts": float(r["points_scored"] or 0)}
        key = (r["fantasy_team"], week_number(r["week"]))
        grouped[key]["start" if r["was_started"] else "bench"].append(p)
    for g in grouped.values():
        g["start"] = _clean_starters(g["start"])
    return grouped


def _position_medians(rosters):
    """{(week_num, pos): median points among started players league-wide}."""
    pts = defaultdict(list)
    for (_, week), g in rosters.items():
        for p in g["start"]:
            pts[(week, p["pos"])].append(p["pts"])
    return {k: median(v) for k, v in pts.items()}


def league_edge(year, min_starts=3):
    """Season edge over a typical starter, best first.

    Per player: ``{name, pos, teams, starts, benches, edge_total,
    edge_per_start, avg_started, avg_benched, wins_swung, swing_weeks}``.
    ``wins_swung`` counts wins where the team would have lost with a median
    starter in that slot. Players with fewer than ``min_starts`` starts are
    left out so one big week doesn't top the table; early in the season the
    bar drops to however many weeks have actually been played.
    """
    rosters = _rosters(year)
    if not rosters:
        return []
    games = _game_results(year)
    medians = _position_medians(rosters)

    agg = {}
    for (team, week), g in rosters.items():
        game = games.get((team, week))
        for p in g["start"]:
            d = agg.setdefault(p["id"], {
                "name": p["name"], "pos": p["pos"], "teams": [],
                "starts": 0, "benches": 0, "started_pts": 0.0, "benched_pts": 0.0,
                "edge_total": 0.0, "wins_swung": 0, "swing_weeks": [],
            })
            edge = p["pts"] - medians.get((week, p["pos"]), 0.0)
            d["starts"] += 1
            d["started_pts"] += p["pts"]
            d["edge_total"] += edge
            if team not in d["teams"]:
                d["teams"].append(team)
            if game and game["result"] == "W" and edge > game["margin"] > 0:
                d["wins_swung"] += 1
                d["swing_weeks"].append(week)
        for p in g["bench"]:
            d = agg.setdefault(p["id"], {
                "name": p["name"], "pos": p["pos"], "teams": [],
                "starts": 0, "benches": 0, "started_pts": 0.0, "benched_pts": 0.0,
                "edge_total": 0.0, "wins_swung": 0, "swing_weeks": [],
            })
            d["benches"] += 1
            d["benched_pts"] += p["pts"]
            if team not in d["teams"]:
                d["teams"].append(team)

    max_starts = max((d["starts"] for d in agg.values()), default=0)
    effective_min = max(1, min(min_starts, max_starts))
    out = []
    for d in agg.values():
        if d["starts"] < effective_min:
            continue
        d["edge_per_start"] = d["edge_total"] / d["starts"]
        d["avg_started"] = d["started_pts"] / d["starts"]
        d["avg_benched"] = (d["benched_pts"] / d["benches"]) if d["benches"] else None
        d["swing_weeks"].sort()
        d["team_label"] = " / ".join(d["teams"])
        out.append(d)
    out.sort(key=lambda d: d["edge_total"], reverse=True)
    return out


def costly_benchings(year):
    """Losses a better start/sit call would have won, one row per game.

    Returns ``{'games': [...], 'team_counts': [...]}``. Each game row is
    ``{team, week, lost_by, options: [{name, pos, pts, would_win_by}]}`` where
    every listed bench player would individually have flipped the result by
    displacing the weakest legal starter.
    """
    rosters = _rosters(year)
    games = _game_results(year)
    rows = []
    for (team, week), g in rosters.items():
        game = games.get((team, week))
        if not game or game["result"] != "L":
            continue
        margin = game["margin"]  # negative in a loss
        options = []
        for b in g["bench"]:
            if b["pos"] in FLEX_POSITIONS:
                legal = [p["pts"] for p in g["start"] if p["pos"] in FLEX_POSITIONS]
            else:
                legal = [p["pts"] for p in g["start"] if p["pos"] == b["pos"]]
            if not legal:
                continue
            swing = b["pts"] - min(legal)
            if swing + margin > 0:
                options.append({"name": b["name"], "pos": b["pos"], "pts": b["pts"],
                                "would_win_by": swing + margin})
        if options:
            options.sort(key=lambda o: -o["would_win_by"])
            rows.append({"team": team, "week": week, "lost_by": -margin,
                         "options": options})
    rows.sort(key=lambda r: (r["week"], r["team"]))

    counts = defaultdict(int)
    for r in rows:
        counts[r["team"]] += 1
    team_counts = sorted(({"team": t, "games": n} for t, n in counts.items()),
                         key=lambda x: -x["games"])
    return {"games": rows, "team_counts": team_counts}


def _trade_log(year):
    """Real trades from the collected Yahoo transaction log.

    Returns ``{(player_id, from_owner, to_owner): trade_key}`` where the
    trade_key is shared by every player in the same trade, or an empty dict
    if the log was never collected for this year. Team names are normalised
    to owner names so they match the roster-derived moves.
    """
    owner = {
        m["team_name"]: m["owner_name"]
        for m in TeamOwnerMapping.objects.filter(year=year, is_active=True).values(
            "team_name", "owner_name")
    }
    log = {}
    rows = PlayerTransaction.objects.filter(
        year=year, transaction_type="TRADE").values(
        "player_id", "from_team", "to_team", "transaction_date")
    for t in rows:
        src = owner.get(t["from_team"], t["from_team"])
        dst = owner.get(t["to_team"], t["to_team"])
        if not src or not dst:
            continue
        key = (t["transaction_date"], frozenset((src, dst)))
        log[(t["player_id"], src, dst)] = key
    return log


def roster_moves(year):
    """Mid-season team changes with production after the move.

    Returns ``{'trades': [...], 'pickups': [...]}``. A move's ``edge_after``
    is the player's league-edge summed over weeks started for the new team
    (until he moves again). Trades come from the Yahoo transaction log when
    it's available; without it, moves between the same two teams within one
    week of each other are grouped as a best-guess trade.
    """
    rosters = _rosters(year)
    if not rosters:
        return {"trades": [], "pickups": [], "exact": False}
    medians = _position_medians(rosters)

    # Rebuild each player's week-by-week team history from the rosters.
    history = defaultdict(dict)   # player_id -> {week: team}
    info = {}
    weekly = defaultdict(dict)    # player_id -> {week: (started, pts, pos)}
    for (team, week), g in rosters.items():
        for status in ("start", "bench"):
            for p in g[status]:
                history[p["id"]][week] = team
                info[p["id"]] = {"name": p["name"], "pos": p["pos"]}
                weekly[p["id"]][week] = (status == "start", p["pts"], p["pos"])

    season_start = min(w for (_, w) in rosters)

    moves = []
    for pid, wks in history.items():
        order = sorted(wks)
        # Collapse the week-by-week history into team stints.
        stints = []  # (arrival_week, team)
        prev = None
        for w in order:
            if wks[w] != prev:
                stints.append((w, wks[w]))
                prev = wks[w]
        for i, (arrived, team) in enumerate(stints):
            if i == 0:
                # First stint: only a move if the player showed up after
                # week 1, i.e. was added off waivers or free agency.
                if arrived <= season_start:
                    continue
                from_team = "Waivers"
            else:
                from_team = stints[i - 1][1]
            end = stints[i + 1][0] if i + 1 < len(stints) else None
            starts = pts_after = edge_after = 0
            for w in order:
                if w < arrived or (end is not None and w >= end):
                    continue
                started, pts, pos = weekly[pid][w]
                if started:
                    starts += 1
                    pts_after += pts
                    edge_after += pts - medians.get((w, pos), 0.0)
            moves.append({
                "player_id": pid, "name": info[pid]["name"], "pos": info[pid]["pos"],
                "from_team": from_team, "to_team": team, "week": arrived,
                "starts_after": starts, "pts_after": pts_after,
                "edge_after": edge_after,
            })

    log = _trade_log(year)
    if log:
        groups, used = _group_trades_from_log(moves, log)
    else:
        groups, used = _group_trades_heuristic(moves)

    trades = []
    for group in groups:
        teams = sorted({moves[j]["to_team"] for j in group})
        team_a = teams[0]
        team_b = teams[1] if len(teams) > 1 else moves[group[0]]["from_team"]
        got_a = [moves[j] for j in group if moves[j]["to_team"] == team_a]
        got_b = [moves[j] for j in group if moves[j]["to_team"] == team_b]
        trades.append({
            "week": min(moves[j]["week"] for j in group),
            "team_a": team_a, "team_b": team_b,
            "got_a": got_a, "got_b": got_b,
            "edge_a": sum(x["edge_after"] for x in got_a),
            "edge_b": sum(x["edge_after"] for x in got_b),
        })
    trades.sort(key=lambda t: t["week"])

    pickups = [m for i, m in enumerate(moves) if i not in used]
    pickups.sort(key=lambda m: -m["edge_after"])
    return {"trades": trades, "pickups": pickups, "exact": bool(log)}


def _group_trades_from_log(moves, log):
    """Group moves into trades using the real transaction log."""
    by_trade = defaultdict(list)
    used = set()
    for i, m in enumerate(moves):
        key = log.get((m["player_id"], m["from_team"], m["to_team"]))
        if key is not None:
            by_trade[key].append(i)
            used.add(i)
    return list(by_trade.values()), used


def _group_trades_heuristic(moves):
    """Best-guess trades: reciprocal moves between two teams within a week."""
    groups, used = [], set()
    for i, m in enumerate(moves):
        if i in used or m["from_team"] == "Waivers":
            continue
        partners = [
            j for j, o in enumerate(moves)
            if j != i and j not in used
            and o["from_team"] == m["to_team"] and o["to_team"] == m["from_team"]
            and abs(o["week"] - m["week"]) <= 1
        ]
        if not partners:
            continue
        group = [i] + partners
        # Pull in any same-direction teammates moving in the same window.
        group += [
            j for j, o in enumerate(moves)
            if j not in group and j not in used
            and {o["from_team"], o["to_team"]} == {m["from_team"], m["to_team"]}
            and abs(o["week"] - m["week"]) <= 1
        ]
        used.update(group)
        groups.append(group)
    return groups, used
