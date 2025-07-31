from django.conf import settings
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
from .models import Player, PlayerRoster, PlayerPerformance, PlayerTransaction
import logging

logger = logging.getLogger(__name__)

class YahooFantasyCollector:
    def __init__(self):
        try:
            # Use different OAuth file based on environment
            oauth_file = 'oauth2_prod.json' if not settings.DEBUG else 'oauth2.json'
            self.oauth = OAuth2(None, None, from_file=oauth_file)
            self.gm = yfa.Game(self.oauth, 'nfl')
            self.league_key = settings.YAHOO_FANTASY_CONFIG['LEAGUE_KEY']
        except Exception as e:
            logger.error(f"Failed to initialize Yahoo API: {e}")
            raise
    
    def get_league_data(self, week=None):
        try:
            league = self.gm.to_league(self.league_key)
            
            teams = league.teams()
            rosters = {}
            player_stats = {}
            
            for team_key, team_data in teams.items():
                team_name = team_data['name']
                roster = league.to_team(team_key).roster(week)
                rosters[team_name] = roster
                
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
                transactions = league.transactions(['add', 'drop', 'trade'], 50)
            except:
                transactions = []
            
            return {
                'teams': teams,
                'rosters': rosters,
                'player_stats': player_stats,
                'transactions': transactions
            }
        except Exception as e:
            logger.error(f"Failed to collect league data: {e}")
            raise
    
    def process_and_save_data(self, week, year):
        data = self.get_league_data(week)
        
        logger.info(f"Processing data for Week {week}, {year}")
        logger.info(f"Teams found: {list(data['rosters'].keys())}")
        
        # Process rosters
        for team_name, roster in data['rosters'].items():
            logger.info(f"Processing {len(roster)} players for team: {team_name}")
            for player_data in roster:
                logger.info(f"Processing player: {player_data.get('name', 'Unknown')}")
                player, created = Player.objects.get_or_create(
                    yahoo_player_id=player_data['player_id'],
                    defaults={
                        'name': player_data['name'],
                        'position': player_data['position_type'],
                        'nfl_team': player_data.get('editorial_team_abbr', '')
                    }
                )
                
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
    
    def was_player_started(self, player_id, rosters):
        for team_name, roster in rosters.items():
            for player in roster:
                if player['player_id'] == player_id:
                    return player.get('selected_position') != 'BN'
        return False