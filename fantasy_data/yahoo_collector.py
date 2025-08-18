from django.conf import settings
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
from .models import Player, PlayerRoster, PlayerPerformance, PlayerTransaction
import logging

logger = logging.getLogger(__name__)

class YahooFantasyCollector:
    def __init__(self):
        try:
            import os
            
            # Check if running on Heroku (production)
            if 'DYNO' in os.environ:
                # Production: use environment variables with token file
                logger.info("Using environment variables for OAuth (production)")
                consumer_key = settings.YAHOO_FANTASY_CONFIG['CLIENT_ID']
                consumer_secret = settings.YAHOO_FANTASY_CONFIG['CLIENT_SECRET']
                
                if not consumer_key or not consumer_secret:
                    raise ValueError("Yahoo API credentials not found in environment variables")
                
                # Create OAuth with credentials and specify token file location
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
                    raise Exception("OAuth token expired. Please re-authorize the application through the admin interface.")
                else:
                    logger.error(f"Local OAuth failed - check oauth2_prod.json file")
                raise
            
            self.league_key = settings.YAHOO_FANTASY_CONFIG['LEAGUE_KEY']
            logger.info(f"Using league key: {self.league_key}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Yahoo API: {e}")
            logger.error(f"Debug info - CLIENT_ID exists: {bool(settings.YAHOO_FANTASY_CONFIG.get('CLIENT_ID'))}")
            logger.error(f"Debug info - CLIENT_SECRET exists: {bool(settings.YAHOO_FANTASY_CONFIG.get('CLIENT_SECRET'))}")
            raise
    
    def get_auth_url(self):
        """Get authorization URL for manual OAuth setup"""
        try:
            import os
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
                    roster = league.to_team(team_key).roster(week)
                    rosters[team_name] = roster
                    logger.info(f"Got roster for {team_name}: {len(roster)} players")
                except Exception as roster_error:
                    logger.error(f"Failed to get roster for {team_name}: {roster_error}")
                    continue
                
                # Get player stats for this team
                try:
                    for player in roster:
                        if week:
                            stats = league.player_stats([player['player_id']], week)
                        else:
                            stats = league.player_stats([player['player_id']])
                        player_stats[player['player_id']] = stats
                except Exception as e:
                    logger.warning(f"Failed to get player stats for {team_name}: {e}")
                    # Continue without player stats for now
            
            # Get transactions with required parameters
            try:
                logger.info("Fetching transactions...")
                transactions = league.transactions(['add', 'drop', 'trade'], 50)
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
            
            # Get matchups for the week
            try:
                logger.info(f"Fetching matchups for week {week}...")
                matchups = league.matchups(week)
                logger.info(f"Found {len(matchups)} matchups for week {week}")
            except Exception as e:
                logger.warning(f"Could not get matchups: {e}")
                matchups = []
            
            team_data = []
            teams = league.teams()
            
            # Create matchup lookup
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
                
                # Get roster
                roster = league.to_team(team_key).roster(week)
                
                # Get player stats for this roster
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
                
                # Calculate both actual and projected points
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
                
                # Calculate projected points separately
                projected_points = 0
                for p in roster_with_stats:
                    if p.get('selected_position') != 'BN':  # Only starters
                        proj_pts = p.get('player_projected_points', {})
                        if isinstance(proj_pts, dict):
                            pts = float(proj_pts.get('total', 0))
                        else:
                            pts = float(proj_pts or 0)
                        projected_points += pts
                
                # Use actual points if no projected points available
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
            wr_points = position_totals['WR'] * (2/3)
        if position_counts['RB'] > 2:  # RB in FLEX -> normalize by 2/3  
            rb_points = position_totals['RB'] * (2/3)
        if position_counts['TE'] > 1:  # TE in FLEX -> normalize by 1/2
            te_points = position_totals['TE'] * (1/2)
        
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
        data = self.get_league_data(week)
        
        logger.info(f"Processing data for Week {week}, {year}")
        logger.info(f"Teams found: {list(data['rosters'].keys())}")
        
        # Process rosters
        for team_name, roster in data['rosters'].items():
            logger.info(f"Processing {len(roster)} players for team: {team_name}")
            for player_data in roster:
                logger.info(f"Processing player: {player_data.get('name', 'Unknown')}")
                logger.info(f"Player data keys: {list(player_data.keys())}")
                logger.info(f"Eligible positions: {player_data.get('eligible_positions', [])}")
                
                # Extract position from eligible_positions list
                eligible_positions = player_data.get('eligible_positions', [])
                position = eligible_positions[0] if eligible_positions else player_data.get('position_type', 'Unknown')
                
                # NFL team data - Yahoo API might not provide this in roster calls
                nfl_team = 'N/A'  # Will need to get from player details API call
                
                player, created = Player.objects.get_or_create(
                    yahoo_player_id=player_data['player_id'],
                    defaults={
                        'name': player_data['name'],
                        'position': position,
                        'nfl_team': nfl_team
                    }
                )
                
                # Update existing player data
                if not created:
                    player.name = player_data['name']
                    player.position = position
                    player.nfl_team = nfl_team
                    player.save()
                
                PlayerRoster.objects.update_or_create(
                    player=player,
                    fantasy_team=team_name,
                    week=f"Week {week}",
                    year=year,
                    defaults={
                        'roster_status': 'STARTER' if player_data.get('selected_position') != 'BN' else 'BENCH'
                    }
                )
                
        logger.info(f"Completed processing. Total players in database: {Player.objects.count()}")
        
        # Also collect team performance data
        try:
            self.collect_team_performance_data(week, year)
        except Exception as e:
            logger.warning(f"Failed to collect team performance data: {e}")
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")
        
        # Process player stats if available
        if data.get('player_stats'):
            for player_id, stats in data['player_stats'].items():
                try:
                    player = Player.objects.get(yahoo_player_id=player_id)
                    points = stats.get('points', 0) if stats else 0
                    
                    PlayerPerformance.objects.update_or_create(
                        player=player,
                        week=f"Week {week}",
                        year=year,
                        defaults={
                            'points_scored': points,
                            'fantasy_team': self.get_player_team(player_id, data['rosters']),
                            'was_started': self.was_player_started(player_id, data['rosters'])
                        }
                    )
                except Player.DoesNotExist:
                    logger.warning(f"Player {player_id} not found in database")
        else:
            logger.info("No player stats data available")
    
    def get_player_team(self, player_id, rosters):
        for team_name, roster in rosters.items():
            for player in roster:
                if player['player_id'] == player_id:
                    return team_name
        return ''
    
    def add_player_stats_to_roster(self, league, roster, week):
        """Add player stats/projections to roster data"""
        roster_with_stats = []
        
        for player in roster:
            player_copy = player.copy()
            player_id = player['player_id']
            
            try:
                # Try to get player stats for this week
                stats = league.player_stats([player_id], week)
                if stats and len(stats) > 0:
                    player_stats = stats[0] if isinstance(stats, list) else stats
                    # Debug: Log what stats we got for sample players
                    if player.get('name') in ['Josh Allen', 'Mike Evans']:
                        logger.info(f"Stats for {player.get('name')}: {player_stats}")
                    # Add stats to player data
                    player_copy['player_points'] = player_stats.get('player_points', {})
                    player_copy['player_projected_points'] = player_stats.get('player_projected_points', {})
                else:
                    # No stats available, set to empty
                    player_copy['player_points'] = {}
                    player_copy['player_projected_points'] = {}
            except Exception as e:
                logger.warning(f"Could not get stats for player {player.get('name', player_id)}: {e}")
                player_copy['player_points'] = {}
                player_copy['player_projected_points'] = {}
            
            roster_with_stats.append(player_copy)
        
        return roster_with_stats
    
    def was_player_started(self, player_id, rosters):
        for team_name, roster in rosters.items():
            for player in roster:
                if player['player_id'] == player_id:
                    return player.get('selected_position') != 'BN'
        return False