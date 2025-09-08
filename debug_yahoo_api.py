#!/usr/bin/env python
"""
Debug script to test Yahoo Fantasy API and identify points data issues
Run this from your Django project directory: python debug_yahoo_api.py
"""

import os
import sys
import django
import logging

# Setup Django environment
sys.path.append('/Users/Matt/Documents/Fantasy_Website/Fantasy_Data')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fantasy_app.settings')
django.setup()

from fantasy_data.yahoo_collector import YahooFantasyCollector
from objectpath import Tree

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _first_nonempty_projection_week(team_obj, league, start_week, max_ahead=2):
    """
    Try start_week, then up to `max_ahead` future weeks, returning the first week
    where at least one player has non-empty player_projected_points.
    """
    try_weeks = [start_week + i for i in range(0, max_ahead + 1)]
    for wk in try_weeks:
        try:
            roster = team_obj.roster(wk)
        except Exception as e:
            logger.warning(f"Could not load roster for week {wk}: {e}")
            continue
        any_proj = any((p.get("player_projected_points") or {}).get("total") is not None for p in roster)
        if any_proj:
            return wk, roster
    return None, None

def debug_yahoo_api():
    """Debug Yahoo API to identify points data issues"""
    try:
        logger.info("=== Yahoo Fantasy API Debug Test ===")

        # Initialize collector
        collector = YahooFantasyCollector()
        league = collector.gm.to_league(collector.league_key)

        # (Optional) sanity check OAuth/league
        logger.info("Testing OAuth connection...")
        logger.info("Successfully connected to Yahoo Fantasy API")
        logger.info(f"Using league key: {collector.league_key}")

        # Test basic connection
        teams = league.teams()
        logger.info(f"Successfully connected! Found {len(teams)} teams")

        # Get first team for testing
        first_team_key = list(teams.keys())[0]
        first_team_name = teams[first_team_key]['name']
        logger.info(f"Testing with team: {first_team_name}")

        team_obj = league.to_team(first_team_key)

        # --- Testing Roster Calls ---
        logger.info("\n--- Testing Roster Calls ---")

        # Current roster
        try:
            current_roster = team_obj.roster()
            logger.info(f"✓ Current roster: {len(current_roster)} players")
        except Exception as e:
            logger.error(f"✗ Current roster failed: {e}")
            return

        # Week 1 roster (kept from your original flow)
        try:
            week_to_test = 1
            week1_roster = team_obj.roster(week_to_test)
            logger.info(f"✓ Week {week_to_test} roster: {len(week1_roster)} players")
            roster_to_test = week1_roster
        except Exception as e:
            logger.error(f"✗ Week 1 roster failed: {e}")
            logger.info("Using current roster for testing")
            roster_to_test = current_roster

        # --- Testing Projected Points (from roster) ---
        # Use the NEXT week by default, since projections for a concluded/active week are often empty in Yahoo.
        logger.info("\n--- Testing Projected Points (from roster) ---")
        try:
            cur_wk = None
            try:
                cur_wk = league.current_week()
            except Exception:
                # If current_week() fails for any reason, fall back to 1
                cur_wk = 1

            default_proj_week = (cur_wk or 1) + 1
            chosen_week, proj_roster = _first_nonempty_projection_week(
                team_obj, league, start_week=default_proj_week, max_ahead=2
            )

            if chosen_week is None or not proj_roster:
                logger.warning(
                    "No non-empty projected points found for the next few weeks. "
                    "This commonly happens right after kickoff when Yahoo removes projections "
                    "for the active/past week and hasn't posted next-week projections yet."
                )
                proj_roster = []
                proj_week = default_proj_week
            else:
                proj_week = chosen_week
                logger.info(f"✓ Found projections on Week {proj_week}: {len(proj_roster)} players")

            # Log the first ~10 for readability
            for p in proj_roster[:10]:
                name = p.get("name", f"Player_{p.get('player_id')}")
                pid = p.get("player_id")
                proj = p.get("player_projected_points") or {}
                total_proj = proj.get("total")
                logger.info(f"  {name} (ID {pid}): projected total = {total_proj!r} | raw={proj}")

            # If we didn't find projections, also explain why Week 1 was empty in your previous run.
            if not proj_roster:
                logger.info(
                    f"(Note) If you request Week {cur_wk} or any past/active week, "
                    "Yahoo may return empty `player_projected_points` ({}). "
                    "Try a future week once Yahoo posts projections."
                )

        except Exception as e:
            logger.error(f"✗ Projected points read failed: {e}")
            proj_roster = []
            proj_week = default_proj_week if 'default_proj_week' in locals() else 1

        # --- Testing Player Stats (Actuals) ---
        if roster_to_test:
            logger.info(f"\n--- Testing Player Stats (Actuals) ---")

            # Test first 3 players
            for i, player in enumerate(roster_to_test[:3]):
                player_id = player['player_id']
                player_name = player.get('name', f'Player_{player_id}')
                position = (player.get('eligible_positions') or ['Unknown'])[0]

                logger.info(f"\nPlayer {i+1}: {player_name} ({position})")
                logger.info(f"  Player ID: {player_id}")
                logger.info(f"  Selected Position: {player.get('selected_position', 'N/A')}")

                # Corrected stat calls: use 'week' and 'season' req_type
                test_methods = [
                    (f"Week {week_to_test} stats (actuals)", lambda: league.player_stats([player_id], 'week', week=week_to_test)),
                    ("Season-to-date (actuals)", lambda: league.player_stats([player_id], 'season')),
                ]

                for method_name, method_func in test_methods:
                    try:
                        stats = method_func()
                        if stats:
                            stat_data = stats[0] if isinstance(stats, list) else stats
                            total_points = stat_data.get('total_points')
                            keys_preview = list(stat_data.keys())[:12]
                            logger.info(f"  ✓ {method_name}: total_points={total_points!r}, keys={keys_preview}...")
                            points = stat_data.get('player_points', {})
                            projected = stat_data.get('player_projected_points', {})
                            if points:
                                logger.info(f"    → Points (unlikely in player_stats payload): {points}")
                            if projected:
                                logger.info(f"    → Projected (unlikely in player_stats payload): {projected}")
                            if not points and not projected:
                                logger.info("    → As expected: no player_points/projected in player_stats; use roster() for projections.")
                        else:
                            logger.warning(f"  ⚠ {method_name}: Empty response")
                    except Exception as e:
                        logger.error(f"  ✗ {method_name} failed: {e}")

        # --- Actual vs Projected quick compare ---
        logger.info(f"\n--- Actual vs Projected (Week {proj_week}) ---")
        try:
            proj_map = {p['player_id']: p for p in proj_roster} if proj_roster else {}
            for p in roster_to_test[:3]:
                pid = p['player_id']
                name = p.get("name", f"Player_{pid}")
                actual_list = league.player_stats([pid], 'week', week=proj_week)
                actual_pts = (actual_list[0].get("total_points") if actual_list and isinstance(actual_list, list) else None)
                projected_total = (proj_map.get(pid, {}).get("player_projected_points") or {}).get("total")
                logger.info(f"  {name} (ID {pid}): projected={projected_total!r} | actual_total_points={actual_pts!r}")
            if not proj_roster:
                logger.info("  (No projections map available for comparison.)")
        except Exception as e:
            logger.error(f"✗ Actual vs Projected compare failed: {e}")

        # --- Testing Matchups ---
        logger.info(f"\n--- Testing Matchups ---")
        try:
            mjson = league.matchups(week_to_test)
            t = Tree(mjson)
            cnts = list(t.execute("$..matchups.count")) or []
            mcount = cnts[0] if cnts else 0
            logger.info(f"✓ Week {week_to_test} matchups: {mcount} found")
            sample = list(t.execute("$..matchup[0]"))
            if sample:
                preview_keys = [k for k in sample[0].keys()] if isinstance(sample[0], dict) else []
                logger.info(f"Sample matchup first element keys: {preview_keys}")
        except Exception as e:
            logger.error(f"✗ Week {week_to_test} matchups failed: {e}")

        logger.info("\n=== Debug Test Complete ===")

    except Exception as e:
        logger.error(f"Debug test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_yahoo_api()
