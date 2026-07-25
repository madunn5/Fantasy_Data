"""Season analytics built on top of the prediction engine.

- standings():      win/loss records and points for/against (optionally through a week)
- luck_report():    expected vs actual wins (who's been lucky / unlucky)
- bench_report():   points left on the bench from the start/sit decisions
- season_schedule(): the matchup grid, derived from the opponent column
- simulate_season(): Monte Carlo playoff odds using each team's scoring distribution
"""
import random

from .models import TeamPerformance, PlayerPerformance, ScheduledMatchup, TeamOwnerMapping
from . import predictions
from .predictions import week_number  # views use analytics.week_number

# Reproducible simulations so the page doesn't flicker between identical loads.
_SIM_SEED = 1234567
_DEFAULT_SIMS = 3000
_DEFAULT_PLAYOFF_SPOTS = 8  # league rule: top 8 make the playoffs


def standings(year, through_week=None):
    """Records + points for/against, best record first.

    ``through_week`` (an int) limits to games up to and including that week.
    """
    rows = TeamPerformance.objects.filter(year=year).values(
        "team_name", "week", "result", "total_points", "points_against"
    )
    teams = {}
    for r in rows:
        if through_week is not None and week_number(r["week"]) > through_week:
            continue
        d = teams.setdefault(r["team_name"],
                             {"wins": 0, "losses": 0, "ties": 0, "pf": 0.0, "pa": 0.0, "games": 0})
        res = (r["result"] or "").upper()
        if res == "W":
            d["wins"] += 1
        elif res == "L":
            d["losses"] += 1
        elif res in ("T", "TIE"):
            d["ties"] += 1
        d["pf"] += float(r["total_points"] or 0)
        d["pa"] += float(r["points_against"] or 0)
        d["games"] += 1
    out = [{"team": t, **d} for t, d in teams.items()]
    out.sort(key=lambda x: (x["wins"], x["pf"]), reverse=True)
    return out


def luck_report(year):
    """Two luck measures per team, luckiest (by scoring luck) first.

    Scoring luck (all-play / expected wins): each week a team's expected wins =
    the fraction of the rest of the league it outscored. Summed over the season
    and compared to actual wins, this isolates *schedule* luck — scoring enough
    to win but drawing the week's hot teams — using only actual scores.

    Projection luck (Yahoo): actual wins minus the weeks Yahoo projected the team
    to win (the ``projected_wins`` field). Measures beating Yahoo's expectation.
    """
    rows = TeamPerformance.objects.filter(year=year).values(
        "team_name", "week", "result", "projected_wins", "total_points"
    )
    agg = {}
    by_week = {}  # week_num -> [(team, score), ...]
    for r in rows:
        team = r["team_name"]
        d = agg.setdefault(team, {"games": 0, "actual": 0, "yahoo": 0, "allplay": 0.0})
        d["games"] += 1
        if (r["result"] or "").upper() == "W":
            d["actual"] += 1
        if (r["projected_wins"] or "").upper() == "W":
            d["yahoo"] += 1
        by_week.setdefault(week_number(r["week"]), []).append((team, float(r["total_points"] or 0)))

    # All-play expected wins: fraction of the field each team outscored, per week.
    for entries in by_week.values():
        n = len(entries)
        if n < 2:
            continue
        for team, score in entries:
            beat = sum(1 for t, s in entries if t != team and score > s)
            tied = sum(1 for t, s in entries if t != team and score == s)
            agg[team]["allplay"] += (beat + 0.5 * tied) / (n - 1)

    out = []
    for team, d in agg.items():
        out.append({
            "team": team,
            "games": d["games"],
            "actual_wins": d["actual"],
            "actual_losses": d["games"] - d["actual"],
            # All-play / scoring luck (primary)
            "allplay_expected_wins": d["allplay"],
            "scoring_luck": d["actual"] - d["allplay"],
            # Yahoo projection luck (secondary)
            "yahoo_expected_wins": d["yahoo"],
            "yahoo_expected_losses": d["games"] - d["yahoo"],
            "projection_luck": d["actual"] - d["yahoo"],
        })
    out.sort(key=lambda x: x["scoring_luck"], reverse=True)
    return out


def bench_report(year):
    """Points left on the bench per team, plus each team's biggest sit blunder."""
    rows = PlayerPerformance.objects.filter(year=year).values(
        "fantasy_team", "week", "points_scored", "was_started",
        "player__name", "player__position",
    )
    teams = {}
    for r in rows:
        d = teams.setdefault(r["fantasy_team"], {
            "bench_points": 0.0, "started_points": 0.0, "bench_count": 0, "top_bench": None,
        })
        pts = float(r["points_scored"] or 0)
        if r["was_started"]:
            d["started_points"] += pts
        else:
            d["bench_points"] += pts
            d["bench_count"] += 1
            if d["top_bench"] is None or pts > d["top_bench"]["points"]:
                d["top_bench"] = {
                    "player": r["player__name"], "position": r["player__position"],
                    "points": pts, "week": r["week"],
                }
    out = [{"team": t, **d} for t, d in teams.items()]
    out.sort(key=lambda x: x["bench_points"], reverse=True)
    return out


def season_schedule(year):
    """{week_num: [(team_a, team_b), ...]} of unique matchups.

    Played weeks come from TeamPerformance's opponent column; future weeks come
    from the stored Yahoo schedule (ScheduledMatchup), with team names
    normalised to owner names via TeamOwnerMapping so both sources join up.
    """
    sched, seen = {}, set()

    def add(wn, a, b):
        if wn <= 0 or not a or not b:
            return
        key = (wn, frozenset((a, b)))
        if key in seen:
            return
        seen.add(key)
        sched.setdefault(wn, []).append((a, b))

    for r in TeamPerformance.objects.filter(year=year).values("team_name", "week", "opponent"):
        add(week_number(r["week"]), r["team_name"], r["opponent"])

    owner = {
        m["team_name"]: m["owner_name"]
        for m in TeamOwnerMapping.objects.filter(year=year, is_active=True).values(
            "team_name", "owner_name")
    }
    for m in ScheduledMatchup.objects.filter(year=year).values("week", "team_a", "team_b"):
        add(m["week"], owner.get(m["team_a"], m["team_a"]), owner.get(m["team_b"], m["team_b"]))
    return sched


def simulate_season(year, through_week=None, n_sims=_DEFAULT_SIMS,
                    playoff_spots=_DEFAULT_PLAYOFF_SPOTS):
    """Monte Carlo playoff odds.

    Records are taken as-is through ``through_week``. Each simulated season
    first draws every team's *true* scoring rate from Normal(mu, mu_sigma) —
    we only have an estimate of how good each team is, and that uncertainty
    is wide early in the season and narrow late, which is what keeps week-2
    odds middle-of-the-road instead of calling the season decided. Every later
    matchup is then simulated by drawing weekly scores around that rate.
    ``mu``/``std`` come from games up to ``through_week`` only (no lookahead —
    the odds "as of week N" only know what week N knew). The top
    ``playoff_spots`` teams by (wins, points-for) make the playoffs.
    Returns ``None`` if there's no data.
    """
    sched = season_schedule(year)
    if not sched:
        return None

    weeks = sorted(sched)
    max_week = max(weeks)
    played = [
        week_number(w) for w in
        TeamPerformance.objects.filter(year=year).values_list("week", flat=True).distinct()
    ]
    latest_played = max(played) if played else 0
    if through_week is None:
        through_week = latest_played
    through_week = max(0, min(through_week, max_week))

    stats = predictions.team_stats(year, through_week=through_week)
    if not stats:
        return None

    base = {s["team"]: s for s in standings(year, through_week)}
    teams = list(stats.keys())
    remaining = [(a, b) for w in weeks if w > through_week for (a, b) in sched[w]]

    rng = random.Random(_SIM_SEED)
    playoff_counts = {t: 0 for t in teams}
    rate_sigma = {t: predictions.mu_sigma(stats[t]) for t in teams}

    for _ in range(n_sims):
        wins = {t: base.get(t, {}).get("wins", 0) for t in teams}
        pf = {t: base.get(t, {}).get("pf", 0.0) for t in teams}
        # One draw of each team's true scoring rate for this simulated season.
        rate = {t: rng.gauss(stats[t]["mu"], rate_sigma[t]) for t in teams}
        for a, b in remaining:
            if a not in stats or b not in stats:
                continue
            sa = rng.gauss(rate[a], stats[a]["std"] or 1.0)
            sb = rng.gauss(rate[b], stats[b]["std"] or 1.0)
            pf[a] += sa
            pf[b] += sb
            if sa >= sb:
                wins[a] += 1
            else:
                wins[b] += 1
        for t in sorted(teams, key=lambda x: (wins[x], pf[x]), reverse=True)[:playoff_spots]:
            playoff_counts[t] += 1

    results = []
    for t in teams:
        s = base.get(t, {})
        pct = 100.0 * playoff_counts[t] / n_sims
        # A simulation can't prove 100% or 0% while games remain, so don't
        # display it: a spot isn't clinched until the schedule says so.
        if remaining and pct > 99.5:
            pct_label = ">99"
        elif remaining and pct < 0.5:
            pct_label = "<1"
        else:
            pct_label = f"{pct:.1f}"
        results.append({
            "team": t,
            "wins": s.get("wins", 0),
            "losses": s.get("losses", 0),
            "pf": s.get("pf", 0.0),
            "playoff_pct": pct,
            "pct_label": pct_label,
        })
    results.sort(key=lambda x: (x["playoff_pct"], x["wins"], x["pf"]), reverse=True)
    return {
        "through_week": through_week,
        "max_week": max_week,
        "latest_played": latest_played,
        "weeks": weeks,
        "n_sims": n_sims,
        "playoff_spots": playoff_spots,
        "remaining_games": len(remaining),
        "teams": results,
    }
