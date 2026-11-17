"""Load a season's NFL regular-season schedule from ESPN into NFLGame.

The Bucket of Death fetches a week on its own the first time it's shown, so this
is optional; running it once a season means draws never wait on ESPN.

Usage:
    python manage.py collect_nfl_schedule --year 2026
"""
from django.core.management.base import BaseCommand
from django.utils import timezone

from draftgame import nfl_schedule


class Command(BaseCommand):
    help = "Load a season's NFL regular-season schedule from ESPN."

    def add_arguments(self, parser):
        parser.add_argument("--year", type=int, default=timezone.now().year)

    def handle(self, *args, **options):
        year = options["year"]
        total = 0
        for week in range(1, nfl_schedule.REGULAR_SEASON_WEEKS + 1):
            saved = nfl_schedule.store_week(year, week)
            total += saved
            self.stdout.write(f"  week {week}: {saved} games")
        self.stdout.write(self.style.SUCCESS(f"Stored {total} games for {year}"))
