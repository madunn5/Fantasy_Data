"""Draft value: did each pick beat what that draft slot usually produces?

A pick's expectation is the average points per week of players taken around
the same spot at the same position (the 10th through 14th running backs
drafted, say), pooled across every draft on the site and leaving the pick
itself out. Value is the player's actual points per week minus that. Season
totals count the player's whole season, whoever rostered him, so this grades
the drafting rather than the in-season moves. Points per week keeps a season
in progress comparable with finished ones.
"""
from collections import defaultdict

from .models import DraftPick

# Picks within this many spots of the same position rank set the expectation.
RANK_WINDOW = 2


def _picks():
    """Every pick with its points per week and positional draft rank."""
    rows = list(DraftPick.objects.select_related("player").order_by("year", "pick"))
    taken = defaultdict(int)
    out = []
    for p in rows:
        taken[(p.year, p.position)] += 1
        ppw = (p.season_points / p.weeks_counted
               if p.season_points is not None and p.weeks_counted else None)
        out.append({"id": p.id, "year": p.year, "round": p.round, "pick": p.pick,
                    "team": p.team, "name": p.player.name, "pos": p.position,
                    "pos_rank": taken[(p.year, p.position)], "ppw": ppw,
                    "weeks": p.weeks_counted})
    return out


def picks_with_value():
    """Every pick with ``expected`` and ``value`` (points per week)."""
    picks = _picks()
    by_pos = defaultdict(list)
    for p in picks:
        if p["ppw"] is not None:
            by_pos[p["pos"]].append(p)
    for p in picks:
        peers = [q["ppw"] for q in by_pos[p["pos"]]
                 if q["id"] != p["id"] and abs(q["pos_rank"] - p["pos_rank"]) <= RANK_WINDOW]
        p["expected"] = sum(peers) / len(peers) if peers else None
        p["value"] = (p["ppw"] - p["expected"]
                      if p["ppw"] is not None and p["expected"] is not None else None)
    return picks


def draft_report(year):
    """Grades, steals, and busts for one draft, plus all-time drafter totals."""
    picks = picks_with_value()
    season = [p for p in picks if p["year"] == year and p["value"] is not None]

    grades = defaultdict(lambda: {"value": 0.0, "picks": []})
    for p in season:
        grades[p["team"]]["value"] += p["value"]
        grades[p["team"]]["picks"].append(p)
    graded = []
    for team, g in grades.items():
        best = max(g["picks"], key=lambda p: p["value"])
        worst = min(g["picks"], key=lambda p: p["value"])
        graded.append({"team": team, "value": g["value"], "best": best, "worst": worst,
                       "picks": sorted(g["picks"], key=lambda p: p["pick"])})
    graded.sort(key=lambda g: -g["value"])

    all_time = defaultdict(lambda: {"value": 0.0, "drafts": set()})
    for p in picks:
        if p["value"] is not None:
            all_time[p["team"]]["value"] += p["value"]
            all_time[p["team"]]["drafts"].add(p["year"])
    career = sorted(({"team": t, "value": d["value"], "drafts": len(d["drafts"]),
                      "per_draft": d["value"] / len(d["drafts"])}
                     for t, d in all_time.items()), key=lambda c: -c["per_draft"])

    ranked = sorted(season, key=lambda p: -p["value"])
    weeks = max((p["weeks"] for p in season), default=0)
    return {
        "grades": graded,
        "steals": ranked[:10],
        "busts": list(reversed(ranked[-10:])),
        "career": career,
        "in_progress": 0 < weeks < 18,
        "weeks": weeks,
        "years": sorted({p["year"] for p in picks}),
    }
