from django.conf import settings
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
from .models import Player, PlayerRoster, PlayerPerformance, PlayerTransaction
import logging
import os

logger = logging.getLogger(__name__)


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
            league = self.gm.to_league(self.league_key)

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
            league = self.gm.to_league(self.league_key)
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

    def collect_team_performance_data(self, week, year):
        """Collect team performance data to replicate CSV upload functionality"""
        try:
            logger.info(f"Starting data collection for week {week}, year {year}")
            logger.info(f"Connecting to league: {self.league_key}")

            league = self.gm.to_league(self.league_key)
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

            # Create matchup lookup (this block assumes a simplified shape; adjust if needed)
            matchup_data = {}
            for matchup in matchups:
                if 'teams' in matchup:
                    teams_in_matchup = matchup['teams']
                    if len(teams_in_matchup) == 2:
                        team1, team2 = teams_in_matchup
                        team1_key = team1.get('team_key')
                        team2_key = team2.get('team_key')
                        team1_points = float(team1.get('team_points', {}).get('total', 0))
                        team2_points = float(team2.get('team_points', {}).get('total', 0))

                        matchup_data[team1_key] = {
                            'opponent_key': team2_key,
                            'points_for': team1_points,
                            'points_against': team2_points
                        }
                        matchup_data[team2_key] = {
                            'opponent_key': team1_key,
                            'points_for': team2_points,
                            'points_against': team1_points
                        }

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
                matchup_info = matchup_data.get(team_key, {})
                opponent_key = matchup_info.get('opponent_key')
                opponent_name = teams.get(opponent_key, {}).get('name', 'TBD') if opponent_key else 'TBD'
                points_for = matchup_info.get('points_for', 0)
                points_against = matchup_info.get('points_against', 0)

                # Calculate result
                if points_for > 0 and points_against > 0:
                    result = 'W' if points_for > points_against else 'L'
                    margin = points_for - points_against
                else:
                    result = 'TBD'
                    margin = 0

                # Calculate actual total (fallback to summing starters if team_points missing)
                if points_for > 0:
                    total_points = points_for
                else:
                    total_points = 0
                    for p in roster_with_stats:
                        if p.get('selected_position') != 'BN':  # Only starters
                            player_pts = p.get('player_points', {})
                            if isinstance(player_pts, dict):
                                pts = float(player_pts.get('total', 0))
                            else:
                                pts = float(player_pts or 0)
                            total_points += pts

                # Calculate projected points separately (from preserved projections)
                projected_points = 0
                for p in roster_with_stats:
                    if p.get('selected_position') != 'BN':  # Only starters
                        proj_pts = p.get('player_projected_points', {})
                        if isinstance(proj_pts, dict):
                            pts = float(proj_pts.get('total', 0))
                        else:
                            pts = float(proj_pts or 0)
                        projected_points += pts

                # Use actual points if no projected points available (keeps legacy behavior)
                if projected_points == 0:
                    projected_points = total_points

                team_record = {
                    'Team': team_name,
                    'Week': f"Week {week}",
                    'Total': total_points,
                    'Expected Total': projected_points,
                    'Difference': total_points - projected_points,
                    'Points Against': points_against,
                    'Opponent': opponent_name,
                    'Result': result,
                    'Margin of Matchup': margin,
                    'Projected Result': result,
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

    def calculate_position_points(self, roster):
        """Calculate position points with FLEX normalization logic"""
        starters = [p for p in roster if p.get('selected_position') != 'BN']

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
            logger.info(f"Player {player.get('name', 'Unknown')} data keys: {list(player.keys())}")

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
        league = self.gm.to_league(self.league_key)

        # Pull teams
        teams = league.teams()
        logger.info(f"Found {len(teams)} teams for writing Players/Rosters")

        players_created = 0
        players_updated = 0
        rosters_written = 0

        for team_key, team_info in teams.items():
            team_name = team_info.get('name', team_key)
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
                status = 'STARTER' if p.get('selected_position') != 'BN' else 'BENCH'
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
                    return player.get('selected_position') != 'BN'
        return False
