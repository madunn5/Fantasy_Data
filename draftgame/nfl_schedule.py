"""NFL schedule lookups for the Bucket of Death.

Bucket weeks line up with NFL regular-season weeks. Games come from ESPN's
public scoreboard and are stored in NFLGame the first time a week is shown;
opponents don't change once the schedule is out, so each week is fetched once.
"""
import logging
import re
from datetime import datetime

import requests

from django.core.cache import cache
from django.utils import timezone

from .models import NFLGame

logger = logging.getLogger(__name__)

ESPN_SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
                   "?dates={year}&seasontype=2&week={week}")
REGULAR_SEASON_WEEKS = 18

NFL_CLUBS = [
    "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
    "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
    "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
    "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
    "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
    "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders",
]
# "Eagles" -> "Philadelphia Eagles", for special picks like "Any Underdog (Eagles)".
_BY_NICKNAME = {club.rsplit(" ", 1)[1].lower(): club for club in NFL_CLUBS}


def fetch_week(year, week):
    """[(home, away, kickoff)] for one regular-season week, straight from ESPN."""
    url = ESPN_SCOREBOARD.format(year=year, week=week)
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    games = []
    for event in data.get("events", []):
        teams = {c["homeAway"]: c["team"]["displayName"]
                 for c in event["competitions"][0]["competitors"]}
        kickoff = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        games.append((teams["home"], teams["away"], kickoff))
    return games


def store_week(year, week):
    """Fetch and save one week's games. Returns how many were saved."""
    games = fetch_week(year, week)
    for home, away, kickoff in games:
        NFLGame.objects.update_or_create(
            year=year, week=week, home_team=home,
            defaults={"away_team": away, "kickoff": kickoff},
        )
    return len(games)


def ensure_week(year, week):
    """Make sure a week's games are stored, fetching them once if needed.

    A failed fetch is remembered for 10 minutes so a slow or down ESPN doesn't
    stall every page load; the page just shows no opponents meanwhile.
    """
    if NFLGame.objects.filter(year=year, week=week).exists():
        return
    fail_key = f"nfl-schedule-failed-{year}-{week}"
    if cache.get(fail_key):
        return
    try:
        store_week(year, week)
    except Exception:
        logger.warning("Couldn't load NFL week %s of %s from ESPN", week, year, exc_info=True)
        cache.set(fail_key, True, 600)


def week_games(year, week):
    """{club: (opponent, is_home, kickoff)} for every team playing that week."""
    ensure_week(year, week)
    games = {}
    for g in NFLGame.objects.filter(year=year, week=week):
        games[g.home_team] = (g.away_team, True, g.kickoff)
        games[g.away_team] = (g.home_team, False, g.kickoff)
    return games


def club_for(team_name):
    """The NFL club a bucket pick is tied to, or None.

    Plain picks are the club itself. Special picks name a club in parentheses
    ("Any Underdog (Eagles)", "The Wizard of Oz (Lions, Tigers, Bears) (Lions)").
    Golden Helmet, Instant Death and picks like Army Football have no club.
    """
    if team_name in NFL_CLUBS:
        return team_name
    from .views import SPECIAL_TEAMS  # views imports this module
    base = next((s for s in SPECIAL_TEAMS if team_name.startswith(s)), None)
    if base is None:
        return None
    choice = team_name[len(base):]
    for club in NFL_CLUBS:
        if club.lower() in choice.lower():
            return club
    for word in re.findall(r"[A-Za-z0-9]+", choice):
        if word.lower() in _BY_NICKNAME:
            return _BY_NICKNAME[word.lower()]
    return None


def matchup_label(team_name, games):
    """'at New York Giants, Thu Nov 12', 'Bye week', or None when it doesn't apply.

    ``games`` is week_games() for the week; an empty dict means the schedule
    isn't available, so nothing is shown rather than a false bye.
    """
    club = club_for(team_name)
    if club is None or not games:
        return None
    if club not in games:
        return "Bye week"
    opponent, is_home, kickoff = games[club]
    label = f"{'vs' if is_home else 'at'} {opponent}"
    if kickoff:
        local = timezone.localtime(kickoff)
        label += f", {local.strftime('%a %b')} {local.day}"
    return label
