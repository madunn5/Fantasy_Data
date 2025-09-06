#!/usr/bin/env python
"""
Debug script to test Yahoo Fantasy API and identify points data issues
Run this from your Django project directory: python debug_yahoo_api.py
"""

import os
import sys
import django

# Setup Django environment
sys.path.append('/Users/Matt/Documents/Fantasy_Website/Fantasy_Data')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fantasy_app.settings')
django.setup()

from fantasy_data.yahoo_collector import YahooFantasyCollector
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_yahoo_api():
    """Debug Yahoo API to identify points data issues"""
    try:
        logger.info("=== Yahoo Fantasy API Debug Test ===")
        
        # Initialize collector
        collector = YahooFantasyCollector()
        league = collector.gm.to_league(collector.league_key)
        
        # Test basic connection
        teams = league.teams()
        logger.info(f"Successfully connected! Found {len(teams)} teams")
        
        # Get first team for testing
        first_team_key = list(teams.keys())[0]
        first_team_name = teams[first_team_key]['name']
        logger.info(f"Testing with team: {first_team_name}")
        
        team_obj = league.to_team(first_team_key)
        
        # Test different roster calls
        logger.info("\n--- Testing Roster Calls ---")
        
        # Current roster
        try:
            current_roster = team_obj.roster()
            logger.info(f"✓ Current roster: {len(current_roster)} players")
        except Exception as e:
            logger.error(f"✗ Current roster failed: {e}")
            return
        
        # Week 1 roster
        try:
            week1_roster = team_obj.roster(1)
            logger.info(f"✓ Week 1 roster: {len(week1_roster)} players")
            roster_to_test = week1_roster
        except Exception as e:
            logger.error(f"✗ Week 1 roster failed: {e}")
            logger.info("Using current roster for testing")
            roster_to_test = current_roster
        
        # Test player stats
        if roster_to_test:
            logger.info(f"\n--- Testing Player Stats ---")
            
            # Test first 3 players
            for i, player in enumerate(roster_to_test[:3]):
                player_id = player['player_id']
                player_name = player.get('name', f'Player_{player_id}')
                position = player.get('eligible_positions', ['Unknown'])[0]
                
                logger.info(f"\nPlayer {i+1}: {player_name} ({position})")
                logger.info(f"  Player ID: {player_id}")
                logger.info(f"  Selected Position: {player.get('selected_position', 'N/A')}")
                
                # Test different stat calls
                test_methods = [
                    ("Week 1 stats", lambda: league.player_stats([player_id], 1)),
                    ("Current stats", lambda: league.player_stats([player_id])),
                    ("Season stats", lambda: league.player_stats([player_id], 'season')),
                ]
                
                for method_name, method_func in test_methods:
                    try:
                        stats = method_func()
                        if stats:
                            logger.info(f"  ✓ {method_name}: {stats}")
                            
                            # Check for points data
                            if isinstance(stats, list) and len(stats) > 0:
                                stat_data = stats[0]
                                points = stat_data.get('player_points', {})
                                projected = stat_data.get('player_projected_points', {})
                                
                                if points:
                                    logger.info(f"    → Points found: {points}")
                                if projected:
                                    logger.info(f"    → Projected found: {projected}")
                                if not points and not projected:
                                    logger.warning(f"    → No points data in response")
                        else:
                            logger.warning(f"  ⚠ {method_name}: Empty response")
                    except Exception as e:
                        logger.error(f"  ✗ {method_name} failed: {e}")
        
        # Test matchups
        logger.info(f"\n--- Testing Matchups ---")
        try:
            matchups = league.matchups(1)
            logger.info(f"✓ Week 1 matchups: {len(matchups)} found")
            if matchups:
                logger.info(f"Sample matchup structure: {matchups[0]}")
        except Exception as e:
            logger.error(f"✗ Week 1 matchups failed: {e}")
        
        logger.info("\n=== Debug Test Complete ===")
        
    except Exception as e:
        logger.error(f"Debug test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_yahoo_api()