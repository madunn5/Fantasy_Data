"""Head-to-head win probability and season power ratings.

Replaces the old logistic-regression model, which didn't work because it
(1) used ``total_points`` as a feature while ``total_points`` is exactly what
decides the game (so it "predicted" the answer it was given), (2) scored each
team in isolation and normalised the two numbers instead of ever comparing the
two teams head-to-head, and (3) ignored the opponent entirely.

This model is simple, honest, and interpretable. A team's weekly score is
treated as a draw from a normal distribution, and for a matchup A vs B the
margin A - B is normal with mean ``muA - muB`` and variance
``sigmaA^2 + sigmaB^2``, so::

    P(A beats B) = Phi( (muA - muB) / sqrt(sigmaA^2 + sigmaB^2) )

The scoring rate ``mu`` used by the model is built in two steps:

1. **Recency weighting.** Each game is weighted by ``0.5 ** (age / half_life)``
   with a 5-week half-life, so a score from 5 weeks ago counts half as much as
   last week's. This lets the rating track trades, injuries, and breakouts
   instead of assuming a team is its September self all year.
2. **Shrinkage toward the league mean.** The weighted average is blended with
   the league-average score using an empirical-Bayes weight of
   ``n_eff / (n_eff + k)`` (k = 3 games' worth of prior). Early in the season a
   team's mean is mostly noise; shrinkage keeps a 160-point week 1 from being
   treated as a 160-point team.

``mean``/``net``/records stay raw season numbers for display; ``mu`` is what
predictions and simulations run on. Teams with very few games borrow the
league-average spread so their probabilities aren't wild. All stats accept
``through_week`` so retrospective simulations only see data up to that week
(no peeking at the future).
"""
import math
import re
from statistics import NormalDist

from .models import TeamPerformance

# Below this many games we don't trust a team's own week-to-week variance and
# fall back to the league-average spread.
_MIN_GAMES_FOR_OWN_VARIANCE = 3

# Empirical-Bayes prior weight: the league-average score counts as this many
# games' worth of evidence when estimating a team's scoring rate.
_SHRINK_PRIOR_GAMES = 3.0

# Recency decay: a game this many weeks old carries half the weight of the
# most recent week. Covers trades / injuries changing what a team really is.
_HALF_LIFE_WEEKS = 5.0
_DECAY = 0.5 ** (1.0 / _HALF_LIFE_WEEKS)

_NORMAL = NormalDist()


def week_number(week):
    """'Week 12' -> 12. Returns 0 if it can't be parsed."""
    m = re.search(r"(\d+)", week or "")
    return int(m.group(1)) if m else 0


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _sample_std(xs):
    """Plain sample standard deviation, or 0.0 with fewer than 2 values."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _weighted_scoring(games, latest_week):
    """Recency-weighted mean/std for one team's ``[(week, points), ...]``.

    Returns ``(w_mean, w_std, n_eff)`` where ``n_eff`` is the effective sample
    size ``(sum w)^2 / sum(w^2)`` — equal to the game count when all weights
    are 1, smaller as old games fade. ``w_std`` is None with fewer than 2 games.
    """
    if not games:
        return 0.0, None, 0.0
    weights = [_DECAY ** (latest_week - wk) for wk, _ in games]
    w_sum = sum(weights)
    w_mean = sum(w * pts for w, (_, pts) in zip(weights, games)) / w_sum
    n_eff = w_sum ** 2 / sum(w * w for w in weights)
    if len(games) < 2 or n_eff <= 1.0:
        return w_mean, None, n_eff
    w_var = sum(w * (pts - w_mean) ** 2 for w, (_, pts) in zip(weights, games)) / w_sum
    # Bessel-style correction using the effective sample size.
    w_std = math.sqrt(w_var * n_eff / (n_eff - 1.0))
    return w_mean, w_std, n_eff


def _weekly_scores(year, through_week=None):
    """{team: {'for': [(week, pts), ...], 'against': [...], 'results': [...]}}.

    ``through_week`` (an int) limits to games up to and including that week.
    """
    rows = TeamPerformance.objects.filter(year=year).values(
        "team_name", "week", "total_points", "points_against", "result"
    )
    data = {}
    for r in rows:
        wn = week_number(r["week"])
        if through_week is not None and wn > through_week:
            continue
        d = data.setdefault(r["team_name"], {"for": [], "against": [], "results": []})
        if r["total_points"] is not None:
            d["for"].append((wn, float(r["total_points"])))
        if r["points_against"] is not None:
            d["against"].append(float(r["points_against"]))
        d["results"].append((r["result"] or "").upper())
    return data


def team_stats(year, through_week=None):
    """Per-team scoring summary for a season (optionally only through a week).

    Returns ``{team: {games, mean, recent_mean, mu, std, mean_against, net,
    wins, losses, ties}}`` where:

    - ``mean``, ``mean_against``, ``net`` are raw season averages (display),
    - ``recent_mean`` is the recency-weighted scoring rate,
    - ``mu`` is ``recent_mean`` shrunk toward the league average — the number
      the win probabilities and simulations actually use,
    - ``std`` is the team's recency-weighted weekly spread, falling back to
      the league-average spread for teams with too few games.
    """
    raw = _weekly_scores(year, through_week)
    if not raw:
        return {}

    all_scores = [pts for d in raw.values() for _, pts in d["for"]]
    league_mean = _mean(all_scores)
    latest_week = max((wk for d in raw.values() for wk, _ in d["for"]), default=0)

    weighted = {team: _weighted_scoring(d["for"], latest_week) for team, d in raw.items()}
    own_stds = [w[1] for w in weighted.values() if w[1] is not None]
    if own_stds:
        league_std = sum(own_stds) / len(own_stds)
    else:
        # Week 1: no team has two games yet, so no team has its own spread.
        # Use the spread of all scores across the league as a stand-in;
        # without this the simulator would treat week 1 results as destiny.
        league_std = _sample_std(all_scores)

    stats = {}
    for team, d in raw.items():
        games = len(d["for"])
        w_mean, w_std, n_eff = weighted[team]
        if w_std is None or games < _MIN_GAMES_FOR_OWN_VARIANCE:
            std = league_std or (w_std or 0.0)
        else:
            std = w_std
        k = _SHRINK_PRIOR_GAMES
        mu = (n_eff * w_mean + k * league_mean) / (n_eff + k) if (n_eff + k) > 0 else league_mean
        mean_for = _mean([pts for _, pts in d["for"]])
        stats[team] = {
            "games": games,
            "mean": mean_for,
            "recent_mean": w_mean,
            "mu": mu,
            "n_eff": n_eff,
            "std": std,
            "mean_against": _mean(d["against"]),
            "net": mean_for - _mean(d["against"]),
            "wins": sum(1 for r in d["results"] if r == "W"),
            "losses": sum(1 for r in d["results"] if r == "L"),
            "ties": sum(1 for r in d["results"] if r in ("T", "TIE")),
        }
    return stats


def mu_sigma(team_stat):
    """How uncertain we still are about a team's true scoring rate ``mu``.

    Standard error of the estimated rate: the weekly spread divided by the
    root of the evidence behind it (effective games plus the prior weight).
    Big early in the season, small by the stretch run.
    """
    return team_stat["std"] / math.sqrt(team_stat["n_eff"] + _SHRINK_PRIOR_GAMES)


def win_probability(team_a, team_b, year, stats=None):
    """Probability that ``team_a`` outscores ``team_b`` in a single matchup.

    Runs on each team's ``mu`` (recency-weighted, shrunk scoring rate). The
    margin variance includes both week-to-week scoring noise and the remaining
    uncertainty in each team's estimated rate, so early-season probabilities
    stay appropriately humble. Returns ``None`` if either team has no data,
    otherwise::

        {p_a, p_b, expected_margin, mu_a, mu_b, mean_a, mean_b, std_a, std_b}
    """
    stats = stats if stats is not None else team_stats(year)
    a, b = stats.get(team_a), stats.get(team_b)
    if not a or not b or a["games"] == 0 or b["games"] == 0:
        return None

    mean_diff = a["mu"] - b["mu"]
    variance = (a["std"] ** 2 + b["std"] ** 2
                + mu_sigma(a) ** 2 + mu_sigma(b) ** 2)
    if variance <= 0:
        p_a = 1.0 if mean_diff > 0 else (0.0 if mean_diff < 0 else 0.5)
    else:
        p_a = _NORMAL.cdf(mean_diff / math.sqrt(variance))
    return {
        "p_a": p_a,
        "p_b": 1 - p_a,
        "expected_margin": mean_diff,
        "mu_a": a["mu"], "mu_b": b["mu"],
        "mean_a": a["mean"], "mean_b": b["mean"],
        "std_a": a["std"], "std_b": b["std"],
    }


def power_ratings(year):
    """Season power-ranking leaderboard, strongest team first.

    Teams are ranked by ``mu`` — recency-weighted points per week, shrunk
    toward the league average — the same number that drives the win
    probabilities and playoff odds. ``avg_for``/``avg_against``/``net`` stay
    raw season averages so the table still shows what actually happened.
    """
    rows = [
        {
            "team": team,
            "rank": 0,
            "games": s["games"],
            "rating": s["mu"],
            "recent_mean": s["recent_mean"],
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
