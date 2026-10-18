from django.conf import settings
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
from .models import Player, PlayerRoster, PlayerPerformance, PlayerTransaction
from .yahoo_timeout_handler import TimeoutYHandler
import logging
import os

logger = logging.getLogger(__name__)

# Lineup slots that do NOT count as active starters: bench, injured reserve
# variants, and inactive. An IR stash must not inflate position counts (it
# would wrongly trigger flex normalization) or count as a start.
NON_STARTER_SLOTS = {'BN', 'IR', 'IR-R', 'NA', 'IL', 'IL+'}


def is_starter_slot(selected_position):
    """True if a Yahoo lineup slot is an active starting spot."""
    return (selected_position or 'BN') not in NON_STARTER_SLOTS


# Yahoo transaction_data types -> PlayerTransaction.transaction_type
_TXN_TYPE_MAP = {'trade': 'TRADE', 'add': 'PICKUP', 'drop': 'DROP'}


def parse_transactions(txns):
    """Flatten Yahoo transaction payloads into one dict per player movement.

    Takes the list returned by ``league.transactions()`` and yields
    ``{player_id, player_name, position, transaction_type, from_team,
    to_team, timestamp}``. Kept separate from the API call so it can be
    unit-tested against a canned payload.
    """
    out = []
    for txn in txns:
        try:
            ts = int(txn.get('timestamp') or 0)
        except (TypeError, ValueError):
            ts = 0
        players = txn.get('players') or {}
        for key, entry in players.items():
            if key == 'count' or not isinstance(entry, dict):
                continue
            plist = entry.get('player')
            if not isinstance(plist, list) or len(plist) < 2:
                continue
            # plist[0] is a list of single-key metadata dicts; plist[1] holds
            # the transaction_data (sometimes wrapped in a one-item list).
            meta = {}
            for item in plist[0]:
                if isinstance(item, dict):
                    meta.update(item)
            tdata = plist[1].get('transaction_data') if isinstance(plist[1], dict) else None
            if isinstance(tdata, list):
                tdata = tdata[0] if tdata else None
            if not isinstance(tdata, dict):
                continue
            ttype = _TXN_TYPE_MAP.get((tdata.get('type') or '').lower())
            if ttype is None:
                continue
            name = meta.get('name')
            if isinstance(name, dict):
                name = name.get('full')
            source = tdata.get('source_team_name')
            if not source and tdata.get('source_type') in ('waivers', 'freeagents'):
                source = 'Waivers'
            dest = tdata.get('destination_team_name')
            if not dest and tdata.get('destination_type') in ('waivers', 'freeagents'):
                dest = 'Waivers'
            out.append({
                'player_id': meta.get('player_id'),
                'player_name': name or f"Player_{meta.get('player_id')}",
                'position': meta.get('display_position') or meta.get('position_type') or '',
                'transaction_type': ttype,
                'from_team': source,
                'to_team': dest,
                'from_team_key': tdata.get('source_team_key'),
                'to_team_key': tdata.get('destination_team_key'),
                'timestamp': ts,
            })
    return out


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_matchups(raw):
    """Flatten Yahoo's scoreboard payload (``league.matchups(week)``) into one
    dict per team: ``{team_key, name, points, projected, opponent_key,
    opponent_name, opponent_points, opponent_projected, status, is_tied,
    winner_team_key}``. ``status`` is 'postevent' once the week is final.
    """
    try:
        league = raw['fantasy_content']['league']
        scoreboard = next(x['scoreboard'] for x in league
                          if isinstance(x, dict) and 'scoreboard' in x)
        matchups = scoreboard['0']['matchups']
    except (KeyError, TypeError, StopIteration):
        return []
    out = []
    for mkey, mval in matchups.items():
        if mkey == 'count' or not isinstance(mval, dict):
            continue
        m = mval.get('matchup') or {}
        teams = (m.get('0') or {}).get('teams') or {}
        sides = []
        for tkey, tval in teams.items():
            if tkey == 'count' or not isinstance(tval, dict):
                continue
            t = tval.get('team')
            if not isinstance(t, list) or len(t) < 2:
                continue
            meta = {}
            for item in t[0]:
                if isinstance(item, dict):
                    meta.update(item)
            stats = t[1] if isinstance(t[1], dict) else {}
            sides.append({
                'team_key': meta.get('team_key'),
                'name': meta.get('name'),
                'points': _to_float((stats.get('team_points') or {}).get('total')),
                'projected': _to_float((stats.get('team_projected_points') or {}).get('total')),
            })
        if len(sides) != 2:
            continue
        for me, opp in ((sides[0], sides[1]), (sides[1], sides[0])):
            out.append({
                **me,
                'opponent_key': opp['team_key'],
                'opponent_name': opp['name'],
                'opponent_points': opp['points'],
                'opponent_projected': opp['projected'],
                'status': m.get('status'),
                'is_tied': str(m.get('is_tied')) == '1',
                'winner_team_key': m.get('winner_team_key'),
            })
    return out


def matchup_result(row):
    """'W'/'L'/'T' for a final matchup row from parse_matchups, else 'TBD'."""
    if row.get('status') != 'postevent':
        return 'TBD'
    if row.get('is_tied'):
        return 'T'
    if row.get('winner_team_key'):
        return 'W' if row['winner_team_key'] == row['team_key'] else 'L'
    if row['points'] == row['opponent_points']:
        return 'T'
    return 'W' if row['points'] > row['opponent_points'] else 'L'


def owner_resolver(year):
    """Return ``resolve(team_key, team_name) -> owner name`` for a season.

    TeamOwnerMapping rows can be keyed by the Yahoo team name (how 2025 was
    mapped) or by the Yahoo team key, which stays fixed all season even when
    a manager renames their team. Unmapped teams keep their Yahoo name.
    """
    from .models import TeamOwnerMapping
    mapping = dict(TeamOwnerMapping.objects.filter(year=year, is_active=True)
                   .values_list('team_name', 'owner_name'))

    def resolve(team_key, team_name):
        return mapping.get(team_name) or mapping.get(team_key) or team_name
    return resolve


class YahooFantasyCollector:
    def __init__(self):
        try:
            # Check if running on Heroku (production)
            if 'DYNO' in os.environ:
                # Check if we have complete OAuth JSON in environment variable
                oauth_json = os.environ.get('YAHOO_OAUTH_JSON')
                if oauth_json:
                    logger.info("Using YAHOO_OAUTH_JSON environment variable")
                    import tempfile
                    # Create temporary file with OAuth JSON
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        f.write(oauth_json)
                        temp_file = f.name

                    try:
                        self.oauth = OAuth2(None, None, from_file=temp_file)
                        os.unlink(temp_file)  # Clean up temp file
                        logger.info("Successfully loaded OAuth from JSON")
                    except Exception as e:
                        os.unlink(temp_file)  # Clean up temp file on error
                        logger.error(f"Failed to load OAuth JSON: {e}")
                        raise
                else:
                    # Fallback to individual environment variables
                    logger.info("Using individual environment variables for OAuth")
                    consumer_key = settings.YAHOO_FANTASY_CONFIG['CLIENT_ID']
                    consumer_secret = settings.YAHOO_FANTASY_CONFIG['CLIENT_SECRET']

                    if not consumer_key or not consumer_secret:
                        raise ValueError("Yahoo API credentials not found in environment variables")

                    self.oauth = OAuth2(consumer_key, consumer_secret, base_url='https://api.login.yahoo.com/')

                    # Check if we have stored tokens
                    if hasattr(self.oauth, 'token') and self.oauth.token:
                        logger.info("Using existing OAuth token")
                    else:
                        logger.error("No valid OAuth token found. Manual authorization required.")
                        raise Exception("OAuth token expired or missing. Please re-authorize the application.")

            else:
                # Local: use oauth2_prod.json file
                base_dir = settings.BASE_DIR
                oauth_file = os.path.join(base_dir, 'oauth2_prod.json')
                logger.info(f"Using OAuth file: {oauth_file}")

                if not os.path.exists(oauth_file):
                    logger.error(f"OAuth file not found: {oauth_file}")
                    raise FileNotFoundError(f"OAuth file not found: {oauth_file}")

                self.oauth = OAuth2(None, None, from_file=oauth_file)

            # Test OAuth connection
            try:
                logger.info("Testing OAuth connection...")
                self.gm = yfa.Game(self.oauth, 'nfl')
                logger.info("Successfully connected to Yahoo Fantasy API")
            except Exception as oauth_error:
                logger.error(f"OAuth connection failed: {oauth_error}")
                if 'DYNO' in os.environ:
                    logger.error("Production OAuth failed - token may be expired")
                    raise Exception(
                        "OAuth token expired. Please re-authorize the application through the admin interface.")
                else:
                    logger.error("Local OAuth failed - check oauth2_prod.json file")
                raise

            self.league_key = settings.YAHOO_FANTASY_CONFIG['LEAGUE_KEY']
            logger.info(f"Using league key: {self.league_key}")

        except Exception as e:
            logger.error(f"Failed to initialize Yahoo API: {e}")
            logger.error(f"Debug info - CLIENT_ID exists: {bool(settings.YAHOO_FANTASY_CONFIG.get('CLIENT_ID'))}")
            logger.error(
                f"Debug info - CLIENT_SECRET exists: {bool(settings.YAHOO_FANTASY_CONFIG.get('CLIENT_SECRET'))}")
            raise

    def _league(self):
        """Return a League with a timeout+retry YHandler injected."""
        league = self.gm.to_league(self.league_key)
        league.inject_yhandler(TimeoutYHandler(self.oauth))
        return league

    def get_auth_url(self):
        """Get authorization URL for manual OAuth setup"""
        try:
            if 'DYNO' in os.environ:
                consumer_key = settings.YAHOO_FANTASY_CONFIG['CLIENT_ID']
                consumer_secret = settings.YAHOO_FANTASY_CONFIG['CLIENT_SECRET']
                oauth = OAuth2(consumer_key, consumer_secret, base_url='https://api.login.yahoo.com/')
            else:
                base_dir = settings.BASE_DIR
                oauth_file = os.path.join(base_dir, 'oauth2_prod.json')
                oauth = OAuth2(None, None, from_file=oauth_file)

            # Get authorization URL without triggering interactive prompt
            auth_url = oauth.get_authorization_url()
            return auth_url
        except Exception as e:
            logger.error(f"Failed to get auth URL: {e}")
            return None

    def is_token_valid(self):
        """Check if current OAuth token is valid"""
        try:
            if hasattr(self, 'oauth') and self.oauth:
                # Try a simple API call to test token
                test_gm = yfa.Game(self.oauth, 'nfl')
                test_gm.game_id()
                return True
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
        return False

    def set_verifier(self, verifier_code):
        """Set OAuth verifier code for token exchange"""
        try:
            if 'DYNO' in os.environ:
                consumer_key = settings.YAHOO_FANTASY_CONFIG['CLIENT_ID']
                consumer_secret = settings.YAHOO_FANTASY_CONFIG['CLIENT_SECRET']
                oauth = OAuth2(consumer_key, consumer_secret, base_url='https://api.login.yahoo.com/')
            else:
                base_dir = settings.BASE_DIR
                oauth_file = os.path.join(base_dir, 'oauth2_prod.json')
                oauth = OAuth2(None, None, from_file=oauth_file)

            # Exchange verifier for access token
            oauth.get_access_token(verifier_code)
            logger.info("Successfully exchanged verifier for access token")
            return True
        except Exception as e:
            logger.error(f"Failed to set verifier: {e}")
            return False

    def test_connection(self):
        """Test the Yahoo API connection and return diagnostic info"""
        try:
            logger.info("=== Yahoo API Connection Test ===")

            # Test basic OAuth
            logger.info("Testing OAuth token...")
            if hasattr(self.oauth, 'access_token') or hasattr(self.oauth, 'token_time'):
                logger.info("OAuth token exists and is valid")
            else:
                logger.error("No OAuth token found")
                return False

            # Test game connection
            logger.info("Testing game connection...")
            game_info = self.gm.game_id()
            logger.info(f"Connected to game: {game_info}")

            # Test league connection
            logger.info(f"Testing league connection: {self.league_key}")
            league = self._league()

            # Get basic league info
            league_settings = league.settings()
            logger.info(f"League name: {league_settings.get('name', 'Unknown')}")
            logger.info(f"League season: {league_settings.get('season', 'Unknown')}")

            # Test teams
            teams = league.teams()
            logger.info(f"Found {len(teams)} teams")
            for team_key, team_data in teams.items():
                logger.info(f"  - {team_data.get('name', 'Unknown')}: {team_key}")

            logger.info("=== Connection Test Successful ===")
            return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def get_league_data(self, week=None):
        try:
            logger.info(f"Attempting to connect to league: {self.league_key}")
            league = self._league()
            logger.info("Successfully connected to league")

            logger.info("Fetching teams...")
            teams = league.teams()
            logger.info(f"Found {len(teams)} teams")

            rosters = {}
            player_stats = {}

            for team_key, team_data in teams.items():
                team_name = team_data['name']
                logger.info(f"Processing team: {team_name}")

                try:
                    # Try to get roster with error handling for position_type issues
                    team_obj = league.to_team(team_key)
                    roster = team_obj.roster(week)
                    rosters[team_name] = roster
                    logger.info(f"Got roster for {team_name}: {len(roster)} players")
                except KeyError as ke:
                    if 'position_type' in str(ke):
                        logger.warning(f"Position type data missing for {team_name}, trying alternative approach")
                        try:
                            # Try without week parameter
                            roster = team_obj.roster()
                            rosters[team_name] = roster
                            logger.info(f"Got roster for {team_name} (no week): {len(roster)} players")
                        except Exception as alt_error:
                            logger.error(f"Alternative roster fetch failed for {team_name}: {alt_error}")
                            rosters[team_name] = []
                    else:
                        logger.error(f"KeyError getting roster for {team_name}: {ke}")
                        rosters[team_name] = []
                except Exception as roster_error:
                    logger.error(f"Failed to get roster for {team_name}: {roster_error}")
                    rosters[team_name] = []

                # Get player stats for this team (FIXED req_type usage)
                try:
                    for player in roster:
                        if week:
                            stats = league.player_stats([player['player_id']], 'week', week=week)
                        else:
                            stats = league.player_stats([player['player_id']], 'season')
                        player_stats[player['player_id']] = stats
                except Exception as e:
                    logger.warning(f"Failed to get player stats for {team_name}: {e}")
                    # Continue without player stats for now

            # Get transactions with required parameters (FIXED CSV + count type)
            try:
                logger.info("Fetching transactions...")
                transactions = league.transactions('add,drop,trade', '50')
                logger.info(f"Found {len(transactions)} transactions")
            except Exception as trans_error:
                logger.warning(f"Failed to get transactions: {trans_error}")
                transactions = []

            return {
                'teams': teams,
                'rosters': rosters,
                'player_stats': player_stats,
                'transactions': transactions
            }
        except Exception as e:
            logger.error(f"Failed to collect league data: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise

    def collect_season_schedule(self, year, start_week=1, end_week=None):
        """Store the league schedule (including future weeks) in ScheduledMatchup.

        TeamPerformance only ever has played weeks, so this is what lets the
        playoff simulator project the rest of the season mid-year. Names are
        stored as Yahoo reports them; season_schedule() maps them to owner
        names via TeamOwnerMapping when reading.
        """
        from .models import ScheduledMatchup

        league = self._league()
        if end_week is None:
            try:
                end_week = int(league.settings().get('playoff_start_week', 15)) - 1
            except Exception as e:
                logger.warning(f"Could not read playoff_start_week, assuming 14 weeks: {e}")
                end_week = 14

        resolve = owner_resolver(year)
        saved = 0
        for week in range(start_week, end_week + 1):
            try:
                rows = parse_matchups(league.matchups(week))
            except Exception as e:
                logger.warning(f"Could not get schedule matchups for week {week}: {e}")
                continue
            for row in rows:
                # Each matchup appears once per side; keep one.
                if not row['team_key'] or not row['opponent_key'] or row['team_key'] > row['opponent_key']:
                    continue
                a, b = sorted((resolve(row['team_key'], row['name']),
                               resolve(row['opponent_key'], row['opponent_name'])))
                ScheduledMatchup.objects.update_or_create(year=year, week=week, team_a=a, team_b=b)
                saved += 1

        logger.info(f"Stored {saved} scheduled matchups for {year} (weeks {start_week}-{end_week})")
        return saved

    def save_upcoming_predictions(self, year, week):
        """Save live predictions for a week whose games haven't been played.

        Uses Yahoo's current projected scores and the model through the week
        before. Weeks already final are skipped, so backfills of old seasons
        never create "live" rows.
        """
        from .matchup_predictions import save_prediction
        from .predictions import team_stats

        league = self._league()
        try:
            end_week = int(league.settings().get('playoff_start_week', 15)) - 1
        except Exception:
            end_week = 14
        if week > end_week:
            return 0
        resolve = owner_resolver(year)
        stats = team_stats(year, through_week=week - 1)
        saved = 0
        for row in parse_matchups(league.matchups(week)):
            if row['status'] == 'postevent' or not row['opponent_key']:
                continue
            if row['team_key'] > row['opponent_key']:
                continue  # each game once
            save_prediction(year, week,
                            resolve(row['team_key'], row['name']),
                            resolve(row['opponent_key'], row['opponent_name']),
                            row['projected'] or None, row['opponent_projected'] or None,
                            'live', stats)
            saved += 1
        logger.info(f"Saved {saved} live predictions for {year} week {week}")
        return saved

    def collect_draft(self, year, weeks_counted=18):
        """Store the season's draft picks with each player's season points.

        Season points are the player's total in league scoring, whoever
        rostered him, so the pick is judged on the player's whole season.
        Re-run during a season to refresh points; ``weeks_counted`` is the
        number of NFL weeks those points cover.
        """
        from .models import DraftPick

        league = self._league()
        resolve = owner_resolver(year)
        picks = league.draft_results()
        ids = [int(p['player_id']) for p in picks if p.get('player_id')]
        stats, details = {}, {}
        for i in range(0, len(ids), 25):
            chunk = ids[i:i + 25]
            for row in league.player_stats(chunk, 'season'):
                stats[int(row['player_id'])] = row
            for row in league.player_details(chunk):
                details[int(row['player_id'])] = row
        saved = 0
        for p in picks:
            pid = p.get('player_id')
            if not pid:
                continue
            pid = int(pid)
            st, det = stats.get(pid, {}), details.get(pid, {})
            position = det.get('primary_position') or det.get('display_position') or 'Unknown'
            if position == 'DEF' or (det.get('position_type') == 'DT'):
                position = 'DEF'
            name = st.get('name') or (det.get('name') or {}).get('full') or f"Player_{pid}"
            player, _ = Player.objects.get_or_create(
                yahoo_player_id=str(pid),
                defaults={'name': name, 'position': position, 'nfl_team': 'N/A'})
            try:
                points = float(st.get('total_points'))
            except (TypeError, ValueError):
                points = None
            DraftPick.objects.update_or_create(
                year=year, pick=int(p['pick']),
                defaults={'round': int(p['round']),
                          'team': resolve(p['team_key'], p['team_key']),
                          'player': player, 'position': position,
                          'season_points': points, 'weeks_counted': weeks_counted})
            saved += 1
        logger.info(f"Stored {saved} draft picks for {year}")
        return saved

    def collect_transactions(self, year, count=100):
        """Pull the league's transaction log into PlayerTransaction rows.

        Records adds, drops, and trades with Yahoo's own type and timestamp,
        so the Player Impact page can label trades exactly instead of
        guessing from roster changes. Safe to re-run: rows are upserted.
        Pass ``count=None`` to fetch the full season log (for backfill).
        """
        from datetime import datetime, timezone as dt_tz
        from .models import Player, PlayerTransaction

        league = self._league()
        txns = league.transactions('add,drop,trade', str(count) if count else '')
        resolve = owner_resolver(year)
        saved = 0
        for rec in parse_transactions(txns):
            if not rec['player_id']:
                continue
            from_team = resolve(rec['from_team_key'], rec['from_team'])
            to_team = resolve(rec['to_team_key'], rec['to_team'])
            player, _ = Player.objects.get_or_create(
                yahoo_player_id=str(rec['player_id']),
                defaults={'name': rec['player_name'],
                          'position': rec['position'] or 'Unknown',
                          'nfl_team': 'N/A'},
            )
            when = datetime.fromtimestamp(rec['timestamp'], tz=dt_tz.utc)
            PlayerTransaction.objects.update_or_create(
                player=player,
                transaction_type=rec['transaction_type'],
                transaction_date=when,
                year=year,
                defaults={'from_team': from_team, 'to_team': to_team},
            )
            saved += 1
        logger.info(f"Stored {saved} transaction rows for {year}")
        return saved

    def collect_team_performance_data(self, week, year):
        """Collect team performance data to replicate CSV upload functionality"""
        try:
            logger.info(f"Starting data collection for week {week}, year {year}")
            logger.info(f"Connecting to league: {self.league_key}")

            league = self._league()
            logger.info("Successfully connected to league for team performance data")

            # Validate week parameter - ensure it's reasonable for current season
            if week < 1 or week > 18:
                logger.warning(f"Week {week} seems invalid, using week 1 as fallback")
                week = 1

            # Get matchups for the requested week (FIXED)
            try:
                logger.info(f"Fetching matchups for week {week}...")
                matchups = league.matchups(week)
                logger.info("Found matchups")
            except Exception as e:
                logger.warning(f"Could not get matchups for week {week}: {e}")
                matchups = []

            team_data = []
            teams = league.teams()

            matchup_data = {row['team_key']: row for row in parse_matchups(matchups)}
            resolve = owner_resolver(year)

            for team_key, team_info in teams.items():
                team_name = team_info['name']

                # Get roster with fallback handling
                try:
                    roster = league.to_team(team_key).roster(week)
                    logger.debug(f"Got roster for {team_name} week {week}: {len(roster)} players")
                except Exception as roster_error:
                    logger.warning(f"Week {week} roster failed for {team_name}: {roster_error}")
                    try:
                        roster = league.to_team(team_key).roster()
                        logger.debug(f"Got current roster for {team_name}: {len(roster)} players")
                    except Exception as current_roster_error:
                        logger.error(f"Could not get any roster for {team_name}: {current_roster_error}")
                        roster = []

                # Get player stats for this roster (adds weekly actuals; preserves projections)
                roster_with_stats = self.add_player_stats_to_roster(league, roster, week)
                position_points = self.calculate_position_points(roster_with_stats)

                # Get matchup info
                matchup_info = matchup_data.get(team_key)
                if not matchup_info or matchup_info['status'] != 'postevent':
                    # Partial scores from an unfinished week would skew the
                    # ratings and luck report, so only final weeks are saved.
                    logger.warning(f"Week {week} not final for {team_name}; skipping team row")
                    continue
                opponent_key = matchup_info['opponent_key']
                opponent_name = resolve(opponent_key, teams.get(opponent_key, {}).get('name')
                                        or matchup_info['opponent_name'])
                points_for = matchup_info['points']
                points_against = matchup_info['opponent_points']
                result = matchup_result(matchup_info)
                margin = points_for - points_against

                # Calculate actual total (fallback to summing starters if team_points missing)
                if points_for > 0:
                    total_points = points_for
                else:
                    total_points = 0
                    for p in roster_with_stats:
                        if is_starter_slot(p.get('selected_position')):
                            player_pts = p.get('player_points', {})
                            if isinstance(player_pts, dict):
                                pts = float(player_pts.get('total', 0))
                            else:
                                pts = float(player_pts or 0)
                            total_points += pts

                # Calculate projected points separately (from preserved projections)
                projected_points = 0
                for p in roster_with_stats:
                    if is_starter_slot(p.get('selected_position')):
                        proj_pts = p.get('player_projected_points', {})
                        if isinstance(proj_pts, dict):
                            pts = float(proj_pts.get('total', 0))
                        else:
                            pts = float(proj_pts or 0)
                        projected_points += pts

                # Yahoo's team projection is the pre-game number, so prefer it.
                if matchup_info['projected'] > 0:
                    projected_points = matchup_info['projected']
                elif projected_points == 0:
                    projected_points = total_points

                opp_projected = matchup_info['opponent_projected']
                if projected_points > 0 and opp_projected > 0:
                    if projected_points == opp_projected:
                        projected_result = 'T'
                    else:
                        projected_result = 'W' if projected_points > opp_projected else 'L'
                else:
                    projected_result = result

                team_record = {
                    'Team': resolve(team_key, team_name),
                    'Week': f"Week {week}",
                    'Total': total_points,
                    'Expected Total': projected_points,
                    'Difference': total_points - projected_points,
                    'Points Against': points_against,
                    'Opponent': opponent_name,
                    'Result': result,
                    'Margin of Matchup': margin,
                    'Projected Result': projected_result,
                    **position_points
                }

                team_data.append(team_record)

            # Save to TeamPerformance model
            from .models import TeamPerformance
            for record in team_data:
                TeamPerformance.objects.update_or_create(
                    team_name=record['Team'],
                    week=record['Week'],
                    year=year,
                    defaults={
                        'qb_points': record['QB_Points'],
                        'wr_points': record['WR_Points'],
                        'wr_points_total': record['WR_Points_Total'],
                        'rb_points': record['RB_Points'],
                        'rb_points_total': record['RB_Points_Total'],
                        'te_points': record['TE_Points'],
                        'te_points_total': record['TE_Points_Total'],
                        'k_points': record['K_Points'],
                        'def_points': record['DEF_Points'],
                        'total_points': record['Total'],
                        'expected_total': record['Expected Total'],
                        'difference': record['Difference'],
                        'points_against': record['Points Against'],
                        'projected_wins': record['Projected Result'],
                        'result': record['Result'],
                        'opponent': record['Opponent'],
                        'margin': record['Margin of Matchup']
                    }
                )

            logger.info(f"Successfully collected team performance data for Week {week}, {year}")
            return team_data

        except Exception as e:
            logger.error(f"Failed to collect team performance data: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    @staticmethod
    def calculate_position_points(roster):
        """Calculate position points with FLEX normalization logic"""
        starters = [p for p in roster if is_starter_slot(p.get('selected_position'))]

        # Group by actual position (not selected_position)
        position_totals = {'QB': 0, 'WR': 0, 'RB': 0, 'TE': 0, 'K': 0, 'DEF': 0}
        position_counts = {'QB': 0, 'WR': 0, 'RB': 0, 'TE': 0, 'K': 0, 'DEF': 0}

        for player in starters:
            # Get primary position from eligible_positions
            eligible_pos = player.get('eligible_positions', [])
            position = eligible_pos[0] if eligible_pos else 'UNKNOWN'

            # Handle D/ST position
            if position == 'D/ST':
                position = 'DEF'

            # Debug: Log available player data keys
            logger.debug(f"Player {player.get('name', 'Unknown')} data keys: {list(player.keys())}")

            # Try actual points first, then projected points
            player_points = player.get('player_points', {})
            projected_points = player.get('player_projected_points', {})

            if isinstance(player_points, dict):
                points = float(player_points.get('total', 0))
            else:
                points = float(player_points or 0)

            # If no actual points, use projected points
            if points == 0:
                if isinstance(projected_points, dict):
                    points = float(projected_points.get('total', 0))
                else:
                    points = float(projected_points or 0)

            # Debug: Log points calculation and data structure for key positions
            if position in ['QB', 'WR'] and points == 0:
                logger.info(f"Player {player.get('name', 'Unknown')} - Position: {position}")
                logger.info(f"  player_points: {player_points}")
                logger.info(f"  player_projected_points: {projected_points}")
                logger.info(f"  Final points: {points}")

            if position in position_totals:
                position_totals[position] += points
                position_counts[position] += 1

        # Apply FLEX normalization (user's specific logic)
        wr_points = position_totals['WR']
        rb_points = position_totals['RB']
        te_points = position_totals['TE']

        # FLEX normalization: if more than standard starters, normalize
        if position_counts['WR'] > 2:  # WR in FLEX -> normalize by 2/3
            wr_points = position_totals['WR'] * (2 / 3)
        if position_counts['RB'] > 2:  # RB in FLEX -> normalize by 2/3
            rb_points = position_totals['RB'] * (2 / 3)
        if position_counts['TE'] > 1:  # TE in FLEX -> normalize by 1/2
            te_points = position_totals['TE'] * (1 / 2)

        return {
            'QB_Points': position_totals['QB'],
            'WR_Points': wr_points,
            'WR_Points_Total': position_totals['WR'],
            'RB_Points': rb_points,
            'RB_Points_Total': position_totals['RB'],
            'TE_Points': te_points,
            'TE_Points_Total': position_totals['TE'],
            'K_Points': position_totals['K'],
            'DEF_Points': position_totals['DEF']
        }

    def process_and_save_data(self, week, year):
        """Collect rosters directly (without get_league_data) and write Players/Rosters,
        then compute & save TeamPerformance."""
        logger.info(f"Processing data for Week {week}, {year} (direct roster fetch)")
        league = self._league()

        # Pull teams
        teams = league.teams()
        logger.info(f"Found {len(teams)} teams for writing Players/Rosters")

        players_created = 0
        players_updated = 0
        rosters_written = 0

        resolve = owner_resolver(year)
        for team_key, team_info in teams.items():
            team_name = resolve(team_key, team_info.get('name', team_key))
            logger.info(f"Fetching roster for team: {team_name} ({team_key})")

            # Try week-specific roster; fall back to current if needed
            try:
                roster = league.to_team(team_key).roster(week)
            except Exception as e:
                logger.warning(
                    f"Week roster failed for {team_name} (week={week}): {e}. Falling back to current roster.")
                try:
                    roster = league.to_team(team_key).roster()
                except Exception as e2:
                    logger.error(f"Could not fetch ANY roster for {team_name}: {e2}")
                    roster = []

            logger.info(f"Got {len(roster)} players for {team_name}")

            # Write players & roster rows
            for p in roster:
                eligible_positions = p.get('eligible_positions') or []
                position = eligible_positions[0] if eligible_positions else (p.get('position_type') or 'Unknown')
                nfl_team = 'N/A'  # not present on roster payload; can enrich later

                # Upsert Player
                player, created = Player.objects.get_or_create(
                    yahoo_player_id=p['player_id'],
                    defaults={
                        'name': p.get('name', 'Unknown'),
                        'position': position,
                        'nfl_team': nfl_team,
                    },
                )
                if created:
                    players_created += 1
                else:
                    updated = False
                    new_name = p.get('name', player.name)
                    if player.name != new_name:
                        player.name = new_name
                        updated = True
                    if player.position != position:
                        player.position = position
                        updated = True
                    if player.nfl_team != nfl_team:
                        player.nfl_team = nfl_team
                        updated = True
                    if updated:
                        player.save()
                        players_updated += 1

                # Upsert PlayerRoster
                status = 'STARTER' if is_starter_slot(p.get('selected_position')) else 'BENCH'
                _, pr_created = PlayerRoster.objects.update_or_create(
                    player=player,
                    fantasy_team=team_name,
                    week=f"Week {week}",
                    year=year,
                    defaults={'roster_status': status},
                )
                if pr_created:
                    rosters_written += 1

                # Create PlayerPerformance record with **weekly** points data (FIXED)
                try:
                    stats = league.player_stats([p['player_id']], 'week', week=week)
                    points_scored = 0.0
                    if stats and len(stats) > 0:
                        points_scored = float(stats[0].get('total_points', 0) or 0)

                    PlayerPerformance.objects.update_or_create(
                        player=player,
                        week=f"Week {week}",
                        year=year,
                        defaults={
                            'fantasy_team': team_name,
                            'points_scored': points_scored,
                            'was_started': (status == 'STARTER')
                        }
                    )
                except Exception as perf_error:
                    logger.warning(f"Could not create PlayerPerformance for {player.name}: {perf_error}")

        logger.info(
            f"Players created: {players_created}, updated: {players_updated}; "
            f"PlayerRoster rows written: {rosters_written}"
        )

        # Always compute & save TeamPerformance too
        try:
            team_data = self.collect_team_performance_data(week, year)
            logger.info(f"TeamPerformance records processed: {len(team_data)}")
        except Exception as e:
            logger.warning(f"collect_team_performance_data failed: {e}")
            team_data = []

        # Keep the stored season schedule fresh so the playoff simulator knows
        # the remaining matchups: full pull the first time, only the current
        # week onward after that (future matchups can still change).
        try:
            from .models import ScheduledMatchup
            start = 1 if not ScheduledMatchup.objects.filter(year=year).exists() else week
            self.collect_season_schedule(year, start_week=start)
        except Exception as e:
            logger.warning(f"collect_season_schedule failed: {e}")

        # Keep the transaction log fresh so trades are labelled exactly. Pull
        # the whole season: a "latest 100" window silently dropped early
        # moves (a Sep 22 2026 trade went missing that way).
        try:
            self.collect_transactions(year, count=None)
        except Exception as e:
            logger.warning(f"collect_transactions failed: {e}")

        # Player projections for start/sit grades. Sleeper serves past weeks,
        # so pulling them after the games is fine; a Sleeper outage shouldn't
        # break the Yahoo collection.
        try:
            from .projections import collect_projections
            rostered = Player.objects.filter(
                id__in=PlayerPerformance.objects.filter(year=year, week=f"Week {week}")
                .values('player_id')).distinct()
            collect_projections(year, [week], players=rostered)
        except Exception as e:
            logger.warning(f"collect_projections failed: {e}")

        # Refresh the current season's draft points so draft value stays
        # current. Past seasons are loaded once (collect_draft command) since
        # their season totals are final.
        if year == settings.YAHOO_FANTASY_CONFIG.get('SEASON'):
            try:
                self.collect_draft(year, weeks_counted=week)
            except Exception as e:
                logger.warning(f"collect_draft failed: {e}")

        # Fill any finished week that never got a live prediction (as a
        # backtest), then save next week's before its games are played.
        try:
            from .matchup_predictions import backtest_season
            backtest_season(year)
            self.save_upcoming_predictions(year, week + 1)
        except Exception as e:
            logger.warning(f"save_upcoming_predictions failed: {e}")

        # Count PlayerPerformance records
        perf_count = PlayerPerformance.objects.filter(week=f"Week {week}", year=year).count()

        return {
            'players_created': players_created,
            'players_updated': players_updated,
            'rosters_written': rosters_written,
            'team_perf_count': len(team_data),
            'player_performances': perf_count,
        }

    def get_player_team(self, player_id, rosters):
        for team_name, roster in rosters.items():
            for player in roster:
                if player['player_id'] == player_id:
                    return team_name
        return ''

    def add_player_stats_to_roster(self, league, roster, week):
        """Add player stats/projections to roster data with corrected API calls"""
        roster_with_stats = []

        for player in roster:
            player_copy = player.copy()
            player_id = player['player_id']
            player_name = player.get('name', f'Player_{player_id}')

            # Preserve projections from roster payload (DON'T overwrite)
            existing_proj = player.get('player_projected_points') or {}
            player_copy['player_projected_points'] = existing_proj

            # Initialize actuals container
            player_copy['player_points'] = {'total': 0}

            try:
                # Weekly actuals if week provided; else season
                if week:
                    stats = league.player_stats([player_id], 'week', week=week)
                else:
                    stats = league.player_stats([player_id], 'season')

                if stats and len(stats) > 0:
                    player_stats = stats[0] if isinstance(stats, list) else stats
                    total_points = float(player_stats.get('total_points', 0) or 0)
                    player_copy['player_points'] = {'total': total_points}
                    if total_points > 0:
                        logger.debug(f"Got {total_points} points for {player_name}")
                else:
                    logger.debug(f"No stats available for {player_name}")

            except Exception as e:
                logger.debug(f"Failed to get stats for {player_name}: {e}")

            roster_with_stats.append(player_copy)

        return roster_with_stats

    def was_player_started(self, player_id, rosters):
        for team_name, roster in rosters.items():
            for player in roster:
                if player['player_id'] == player_id:
                    return is_starter_slot(player.get('selected_position'))
        return False
