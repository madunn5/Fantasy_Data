"""Weekly matchup predictions, saved ahead of results, and how they did.

The forecast for each game comes from Yahoo's pre-game projected scores: the
projected margin over each team's week-to-week spread, widened for projection
error. A backtest over 2023-2026 (257 games) settled this: projections picked
58% of winners with the best Brier score (0.241), while the season-scoring
model picked 49%, no better than a coin flip, because past scoring says
little about any one week. The model's number is still saved for comparison.

Live predictions are saved when the weekly collection runs, before that week's
games. Earlier weeks are filled in as backtests that only use information
from before each week.
"""
import math
from collections import defaultdict
from statistics import NormalDist

from .models import MatchupPrediction, TeamPerformance
from .predictions import team_stats, week_number, win_probability

# Projections miss by more than a team's own weekly swings; widening the
# spread by this factor gave the best-calibrated chances in the backtest.
PROJECTION_SPREAD = 1.5
_NORMAL = NormalDist()

# Calibration buckets for "how often did the favorite win" (favorite's chance).
CONFIDENCE_BUCKETS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]


def model_probability(year, week, team_a, team_b, stats=None):
    """P(team_a beats team_b) from games through week - 1, or None in week 1."""
    if week <= 1:
        return None
    stats = stats if stats is not None else team_stats(year, through_week=week - 1)
    prob = win_probability(team_a, team_b, year, stats=stats)
    return prob["p_a"] if prob else None


def projection_probability(proj_a, proj_b, std_a, std_b):
    """P(team_a wins) from projected scores and each team's weekly spread."""
    if not proj_a or not proj_b:
        return None
    spread = PROJECTION_SPREAD * math.sqrt(std_a ** 2 + std_b ** 2)
    if spread <= 0:
        return 0.5 if proj_a == proj_b else float(proj_a > proj_b)
    return _NORMAL.cdf((proj_a - proj_b) / spread)


def _spreads(year, week, stats):
    """Each team's weekly spread before ``week``, falling back to the league's."""
    if not stats:
        stats = team_stats(year)  # week 1: borrow the season's spread shape
    league = [s["std"] for s in stats.values() if s.get("std")]
    fallback = sum(league) / len(league) if league else 25.0
    return lambda team: (stats.get(team) or {}).get("std") or fallback


def save_prediction(year, week, team_a, team_b, proj_a, proj_b, source, stats=None):
    """Upsert one game's prediction with teams in alphabetical order."""
    if team_b < team_a:
        team_a, team_b, proj_a, proj_b = team_b, team_a, proj_b, proj_a
    if stats is None and week > 1:
        stats = team_stats(year, through_week=week - 1)
    spread = _spreads(year, week, stats)
    MatchupPrediction.objects.update_or_create(
        year=year, week=week, team_a=team_a, team_b=team_b,
        defaults={"model_p_a": model_probability(year, week, team_a, team_b, stats),
                  "proj_p_a": projection_probability(proj_a, proj_b,
                                                     spread(team_a), spread(team_b)),
                  "proj_a": proj_a, "proj_b": proj_b, "source": source},
    )


def backtest_season(year):
    """Fill in predictions for every completed week that doesn't have one.

    Uses each week's stored pre-game projections (expected_total) and the
    model as of the week before. Existing rows, including live ones, are
    left alone. Returns the number of games added.
    """
    rows = TeamPerformance.objects.filter(year=year).values(
        "team_name", "week", "opponent", "expected_total")
    by_week = defaultdict(dict)
    for r in rows:
        by_week[week_number(r["week"])][r["team_name"]] = r
    existing = set(MatchupPrediction.objects.filter(year=year).values_list(
        "week", "team_a", "team_b"))
    added = 0
    for week in sorted(by_week):
        teams = by_week[week]
        stats = team_stats(year, through_week=week - 1) if week > 1 else None
        for team, r in teams.items():
            opp = r["opponent"]
            if not opp or opp not in teams or opp < team:
                continue  # each game once, from the alphabetically first side
            if (week, team, opp) in existing:
                continue
            save_prediction(year, week, team, opp,
                            _float(r["expected_total"]), _float(teams[opp]["expected_total"]),
                            "backtest", stats)
            added += 1
    return added


def _float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def _results(year):
    """{(team, week): 'W'/'L'/'T'} for final games."""
    return {(r["team_name"], week_number(r["week"])): (r["result"] or "").upper()
            for r in TeamPerformance.objects.filter(year=year).values("team_name", "week", "result")}


def graded_predictions(year):
    """Every saved prediction for a season with its outcome, by week.

    Each row: ``{week, team_a, team_b, model_p_a, model_pick, model_conf,
    proj_a, proj_b, proj_pick, winner, model_right, proj_right, source}``;
    winner and the *_right flags are None until the game is final (or a tie).
    """
    results = _results(year)
    out = []
    def pick(p_a, a, b):
        if p_a is None or p_a == 0.5:
            return None, None
        return (a if p_a > 0.5 else b), max(p_a, 1 - p_a)

    for p in MatchupPrediction.objects.filter(year=year).order_by("week", "team_a"):
        res = results.get((p.team_a, p.week))
        winner = p.team_a if res == "W" else p.team_b if res == "L" else None
        forecast_p = p.proj_p_a if p.proj_p_a is not None else p.model_p_a
        f_pick, f_conf = pick(forecast_p, p.team_a, p.team_b)
        m_pick, m_conf = pick(p.model_p_a, p.team_a, p.team_b)
        out.append({
            "week": p.week, "team_a": p.team_a, "team_b": p.team_b,
            "proj_a": p.proj_a, "proj_b": p.proj_b, "source": p.source,
            "forecast_p_a": forecast_p, "pick": f_pick, "conf": f_conf,
            "model_p_a": p.model_p_a, "model_pick": m_pick, "model_conf": m_conf,
            "winner": winner,
            "right": (f_pick == winner) if winner and f_pick else None,
            "model_right": (m_pick == winner) if winner and m_pick else None,
        })
    return out


def accuracy_summary(graded):
    """Forecast hit rate, the season-scoring model's for comparison,
    calibration buckets, and week-by-week results."""
    def rate(flags):
        flags = [f for f in flags if f is not None]
        return {"right": sum(flags), "total": len(flags),
                "pct": (100.0 * sum(flags) / len(flags)) if flags else None}

    calibration = []
    for lo, hi in CONFIDENCE_BUCKETS:
        bucket = [g for g in graded if g["right"] is not None and lo <= g["conf"] < hi]
        if bucket:
            calibration.append({
                "label": f"{int(lo * 100)}-{min(int(hi * 100), 100)}%",
                "games": len(bucket),
                "avg_conf": 100.0 * sum(g["conf"] for g in bucket) / len(bucket),
                "won": 100.0 * sum(g["right"] for g in bucket) / len(bucket),
            })
    weeks = defaultdict(list)
    for g in graded:
        weeks[g["week"]].append(g["right"])
    by_week = [{"week": w, "forecast": rate(v)} for w, v in sorted(weeks.items())]
    return {
        "forecast": rate([g["right"] for g in graded]),
        "model": rate([g["model_right"] for g in graded]),
        "calibration": calibration, "by_week": by_week,
    }
