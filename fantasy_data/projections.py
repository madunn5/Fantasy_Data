"""Weekly player projections from Sleeper, stored per Yahoo player.

Yahoo's API only exposes team-level projections, so player projections come
from Sleeper's public API, which also serves past weeks. They are scored as
half-PPR to match the league (0.5 per reception, 4-point passing TDs); kicker
and defense scoring is Sleeper's default, close enough for comparing start/sit
options. Rows land in ProjectedPoint with source='sleeper' and league_key set
to the season year.

Players are matched to our Yahoo players by Sleeper's yahoo_id when present,
otherwise by normalized name and position; defenses map by NFL team.
"""
import logging
import re
from collections import defaultdict

import requests

from .models import Player, ProjectedPoint

logger = logging.getLogger(__name__)

SLEEPER_PROJECTIONS = "https://api.sleeper.app/projections/nfl/{season}/{week}"
SLEEPER_PLAYERS = "https://api.sleeper.app/v1/players/nfl"
SCORING_FIELD = "pts_half_ppr"
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")

# Yahoo defense player ids are 100000 + Yahoo's NFL team id.
YAHOO_DEF_TEAM = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET",
    9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA", 16: "MIN",
    17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC",
    25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name):
    """'Kenneth Walker III' -> 'kenneth walker'; 'Amon-Ra St. Brown' -> 'amon ra st brown'."""
    cleaned = re.sub(r"[^a-z ]", "", (name or "").lower().replace("-", " "))
    return " ".join(w for w in cleaned.split() if w not in _SUFFIXES)


def build_matcher(sleeper_players):
    """Return ``match(player) -> sleeper_id or None`` for our Player rows."""
    by_yahoo = {str(p["yahoo_id"]): sid for sid, p in sleeper_players.items() if p.get("yahoo_id")}
    by_name_pos = defaultdict(set)
    for sid, p in sleeper_players.items():
        name = normalize_name(p.get("full_name")
                              or f"{p.get('first_name', '')} {p.get('last_name', '')}")
        if not name:
            continue
        # fantasy_positions covers two-way players listed elsewhere (a CB/WR).
        for pos in set(p.get("fantasy_positions") or []) | {p.get("position")}:
            if pos in POSITIONS:
                by_name_pos[(name, pos)].add(sid)

    def match(player):
        if player.position == "DEF":
            try:
                return YAHOO_DEF_TEAM.get(int(player.yahoo_player_id) - 100000)
            except (TypeError, ValueError):
                return None
        if str(player.yahoo_player_id) in by_yahoo:
            return by_yahoo[str(player.yahoo_player_id)]
        candidates = by_name_pos.get((normalize_name(player.name), player.position), set())
        return next(iter(candidates)) if len(candidates) == 1 else None
    return match


def fetch_sleeper_players():
    resp = requests.get(SLEEPER_PLAYERS, timeout=120)
    resp.raise_for_status()
    return resp.json()


def fetch_week_projections(season, week):
    """{sleeper_id: projected half-PPR points} for one week."""
    params = [("season_type", "regular")] + [("position[]", p) for p in POSITIONS]
    resp = requests.get(SLEEPER_PROJECTIONS.format(season=season, week=week),
                        params=params, timeout=120)
    resp.raise_for_status()
    out = {}
    for row in resp.json():
        pts = (row.get("stats") or {}).get(SCORING_FIELD)
        if pts is not None:
            out[str(row["player_id"])] = float(pts)
    return out


def collect_projections(season, weeks, players=None, sleeper_players=None):
    """Store projections for ``players`` (default: everyone in Player) for each week.

    Returns ``{'stored': n, 'unmatched': [names]}``. Re-runnable: rows are
    upserted. Players Sleeper has no projection for are left out, which the
    analysis treats as no projection rather than zero.
    """
    sleeper_players = sleeper_players or fetch_sleeper_players()
    match = build_matcher(sleeper_players)
    players = list(players if players is not None else Player.objects.all())
    sleeper_ids = {}
    unmatched = []
    for pl in players:
        sid = match(pl)
        if sid:
            sleeper_ids[pl.yahoo_player_id] = (sid, pl)
        else:
            unmatched.append(f"{pl.name} ({pl.position})")

    stored = 0
    for week in weeks:
        proj = fetch_week_projections(season, week)
        week_count = 0
        for yid, (sid, pl) in sleeper_ids.items():
            if sid not in proj:
                continue
            try:
                yahoo_id = int(yid)
            except (TypeError, ValueError):
                continue
            ProjectedPoint.objects.update_or_create(
                league_key=str(season), player_id=yahoo_id, week=week, source="sleeper",
                defaults={"player_name": pl.name, "position": pl.position,
                          "projected_total": round(proj[sid], 2),
                          "projected_raw": {"sleeper_id": sid, SCORING_FIELD: proj[sid]}},
            )
            week_count += 1
        stored += week_count
        logger.info(f"Sleeper projections {season} week {week}: {week_count} players")
    return {"stored": stored, "unmatched": sorted(set(unmatched))}
