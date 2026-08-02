from django.core.management.base import BaseCommand, CommandError
from fantasy_data.yahoo_collector import YahooFantasyCollector
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Collect data from Yahoo Fantasy API'

    def add_arguments(self, parser):
        parser.add_argument('--week', type=int, help='Week number (required unless --schedule-only)')
        parser.add_argument('--year', type=int, required=True, help='Season year')
        parser.add_argument('--schedule-only', action='store_true',
                            help='Only refresh the stored season schedule (for playoff odds)')
        parser.add_argument('--transactions-only', action='store_true',
                            help='Only pull the transaction log (full season, for backfill)')

    def handle(self, *args, **options):
        week = options['week']
        year = options['year']

        try:
            collector = YahooFantasyCollector()
            if options['schedule_only']:
                saved = collector.collect_season_schedule(year)
                self.stdout.write(
                    self.style.SUCCESS(f'Stored {saved} scheduled matchups for {year}')
                )
                return
            if options['transactions_only']:
                saved = collector.collect_transactions(year, count=None)
                self.stdout.write(
                    self.style.SUCCESS(f'Stored {saved} transaction rows for {year}')
                )
                return
            if week is None:
                raise CommandError('--week is required unless --schedule-only is passed')
            collector.process_and_save_data(week, year)
            self.stdout.write(
                self.style.SUCCESS(f'Successfully collected data for Week {week}, {year}')
            )
        except CommandError:
            raise
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to collect data: {e}')
            )
            logger.error(f'Data collection failed: {e}')
