import time
import logging
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from fantasy_data.yahoo_collector import YahooFantasyCollector
from fantasy_data.models import ProjectedPoint

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Snapshot Yahoo weekly player projections into ProjectedPoint."

    def add_arguments(self, parser):
        parser.add_argument("--week", type=int, default=None, help="NFL week to snapshot (defaults to current_week + 1)")
        parser.add_argument("--max-ahead", type=int, default=0, help="If no projections for --week/default, try up to N weeks ahead")
        parser.add_argument("--sleep", type=float, default=0.35, help="Sleep seconds between team roster calls (throttle)")

    def handle(self, *args, **opts):
        collector = YahooFantasyCollector()
        league = collector.gm.to_league(collector.league_key)
        league_key = collector.league_key

        # Determine target week (default: next week)
        try:
            current_week = league.current_week()
        except Exception:
            current_week = 1

        target_week = opts["week"] or (current_week + 1)
        max_ahead = max(0, int(opts["max_ahead"]))
        sleep_s = float(opts["sleep"])

        # Try target_week, optionally look ahead if empty
        week_candidates = [target_week + i for i in range(max_ahead + 1)]

        # Get all teams
        teams = league.teams()
        team_keys = list(teams.keys())
        self.stdout.write(self.style.SUCCESS(f"Found {len(team_keys)} teams in league {league_key}"))
        total_saved = 0
        chosen_week = None

        for wk in week_candidates:
            self.stdout.write(self.style.NOTICE(f"Attempting to snapshot projections for week {wk}"))
            saved_this_week = 0

            for tkey in team_keys:
                team_obj = league.to_team(tkey)
                try:
                    roster = team_obj.roster(wk)
                except Exception as e:
                    logger.warning(f"Failed to load roster for {tkey} week {wk}: {e}")
                    continue

                for p in roster:
                    pid = p.get("player_id")
                    name = p.get("name") or f"Player_{pid}"
                    pos_list = p.get("eligible_positions") or []
                    pos = pos_list[0] if pos_list else ""
                    proj = p.get("player_projected_points") or {}
                    total_proj = proj.get("total")

                    # Only store when we actually have a projection value
                    if total_proj is None:
                        continue

                    # Upsert
                    obj, created = ProjectedPoint.objects.update_or_create(
                        league_key=league_key,
                        player_id=pid,
                        week=wk,
                        source="yahoo",
                        defaults={
                            "player_name": name,
                            "position": pos,
                            "team_key": tkey,
                            "projected_total": total_proj,
                            "projected_raw": proj,
                            "pulled_at": timezone.now(),
                        },
                    )
                    saved_this_week += 1

                time.sleep(sleep_s)  # polite throttle

            if saved_this_week > 0:
                chosen_week = wk
                total_saved += saved_this_week
                self.stdout.write(self.style.SUCCESS(f"Saved {saved_this_week} projection rows for week {wk}"))
                break  # stop at the first week that has projections
            else:
                self.stdout.write(self.style.WARNING(f"No non-empty projections found for week {wk}"))

        if chosen_week is None:
            raise CommandError(
                f"No projections found for weeks: {week_candidates}. "
                "This is normal immediately after kickoff or before Yahoo posts next-week projections."
            )

        self.stdout.write(self.style.SUCCESS(
            f"Snapshot complete: week {chosen_week}, rows saved: {total_saved}"
        ))