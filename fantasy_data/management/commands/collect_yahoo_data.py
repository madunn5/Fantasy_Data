from django.core.management.base import BaseCommand
from fantasy_data.yahoo_collector import YahooFantasyCollector
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Collect data from Yahoo Fantasy API'
    
    def add_arguments(self, parser):
        parser.add_argument('--week', type=int, required=True, help='Week number')
        parser.add_argument('--year', type=int, required=True, help='Season year')
    
    def handle(self, *args, **options):
        week = options['week']
        year = options['year']
        
        try:
            collector = YahooFantasyCollector()
            collector.process_and_save_data(week, year)
            self.stdout.write(
                self.style.SUCCESS(f'Successfully collected data for Week {week}, {year}')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to collect data: {e}')
            )
            logger.error(f'Data collection failed: {e}')