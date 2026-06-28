"""Head-to-head win probability and season power ratings.

Replaces the old logistic-regression model, which didn't work because it
(1) used ``total_points`` as a feature while ``total_points`` is exactly what
decides the game (so it "predicted" the answer it was given), (2) scored each
team in isolation and normalised the two numbers instead of ever comparing the
two teams head-to-head, and (3) ignored the opponent entirely.

This model is simple, honest, and interpretable. A team's weekly score is
treated as a draw from a normal distribution estimated from its game log
(mean ``mu``, standard deviation ``sigma``). For a matchup A vs B the margin
A - B is normal with mean ``muA - muB`` and variance ``sigmaA^2 + sigmaB^2``,
so::

    P(A beats B) = Phi( (muA - muB) / sqrt(sigmaA^2 + sigmaB^2) )

Evenly matched teams land near 50%, and the projected margin falls straight out
of the same numbers. Teams with very few games borrow the league-average spread
so their probabilities aren't wild.
"""
import math
from statistics import NormalDist

from .models import TeamPerformance

# Below this many games we don't trust a team's own week-to-week variance and
# fall back to the league-average spread.
_MIN_GAMES_FOR_OWN_VARIANCE = 3

_NORMAL = NormalDist()


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    """Sample standard deviation, or None if fewer than 2 values."""
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _weekly_scores(year):
    """{team: {'for': [...], 'against': [...], 'results': [...]}} for a season."""
    rows = TeamPerformance.objects.filter(year=year).values(
        "team_name", "total_points", "points_against", "result"
    )
    data = {}
    for r in rows:
        d = data.setdefault(r["team_name"], {"for": [], "against": [], "results": []})
        if r["total_points"] is not None:
            d["for"].append(float(r["total_points"]))
        if r["points_against"] is not None:
            d["against"].append(float(r["points_against"]))
        d["results"].append((r["result"] or "").upper())
    return data


def team_stats(year):
    """Per-team scoring summary for a season.

    Returns ``{team: {games, mean, std, mean_against, net, wins, losses, ties}}``
    where ``std`` is the team's own weekly spread, falling back to the
    league-average spread for teams with too few games.
    """
    raw = _weekly_scores(year)
    if not raw:
        return {}

    own_stds = [s for s in (_std(d["for"]) for d in raw.values()) if s is not None]
    league_std = (sum(own_stds) / len(own_stds)) if own_stds else 0.0

    stats = {}
    for team, d in raw.items():
        games = len(d["for"])
        own_std = _std(d["for"])
        if own_std is None or games < _MIN_GAMES_FOR_OWN_VARIANCE:
            std = league_std or (own_std or 0.0)
        else:
            std = own_std
        mean_for = _mean(d["for"])
        stats[team] = {
            "games": games,
            "mean": mean_for,
            "std": std,
            "mean_against": _mean(d["against"]),
            "net": mean_for - _mean(d["against"]),
            "wins": sum(1 for r in d["results"] if r == "W"),
            "losses": sum(1 for r in d["results"] if r == "L"),
            "ties": sum(1 for r in d["results"] if r in ("T", "TIE")),
        }
    return stats


def win_probability(team_a, team_b, year, stats=None):
    """Probability that ``team_a`` outscores ``team_b`` in a single matchup.

    Returns ``None`` if either team has no data, otherwise::

        {p_a, p_b, expected_margin, mean_a, mean_b, std_a, std_b}
    """
    stats = stats if stats is not None else team_stats(year)
    a, b = stats.get(team_a), stats.get(team_b)
    if not a or not b or a["games"] == 0 or b["games"] == 0:
        return None

    mean_diff = a["mean"] - b["mean"]
    variance = a["std"] ** 2 + b["std"] ** 2
    if variance <= 0:
        p_a = 1.0 if mean_diff > 0 else (0.0 if mean_diff < 0 else 0.5)
    else:
        p_a = _NORMAL.cdf(mean_diff / math.sqrt(variance))
    return {
        "p_a": p_a,
        "p_b": 1 - p_a,
        "expected_margin": mean_diff,
        "mean_a": a["mean"], "mean_b": b["mean"],
        "std_a": a["std"], "std_b": b["std"],
    }


def power_ratings(year):
    """Season power-ranking leaderboard, strongest team first.

    Teams are ranked by average points scored per week — in fantasy you don't
    control your opponent, so scoring output is the cleanest measure of team
    strength, and it's the same number that drives the win probabilities.
    """
    rows = [
        {
            "team": team,
            "rank": 0,
            "games": s["games"],
            "rating": s["mean"],          # avg points scored / week
            "avg_for": s["mean"],
            "avg_against": s["mean_against"],
            "net": s["net"],
            "std": s["std"],
            "wins": s["wins"], "losses": s["losses"], "ties": s["ties"],
        }
        for team, s in team_stats(year).items()
    ]
    rows.sort(key=lambda r: r["rating"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows
