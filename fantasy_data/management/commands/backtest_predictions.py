from django.core.management.base import BaseCommand

from fantasy_data.matchup_predictions import backtest_season


class Command(BaseCommand):
    help = "Fill in matchup predictions for completed weeks, using only earlier weeks."

    def add_arguments(self, parser):
        parser.add_argument('--year', type=int, action='append', required=True,
                            help='Season to backtest (repeatable)')

    def handle(self, *args, **opts):
        for year in opts['year']:
            added = backtest_season(year)
            self.stdout.write(self.style.SUCCESS(f"{year}: added {added} backtest predictions"))
