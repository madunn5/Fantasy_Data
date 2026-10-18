from django.core.management.base import BaseCommand

from fantasy_data.yahoo_collector import YahooFantasyCollector


class Command(BaseCommand):
    help = ("Store a season's draft picks with each player's season points. "
            "Set YAHOO_LEAGUE_KEY for past seasons.")

    def add_arguments(self, parser):
        parser.add_argument('--year', type=int, required=True)
        parser.add_argument('--weeks', type=int, default=18,
                            help='NFL weeks the season points cover (default 18, a full season)')

    def handle(self, *args, **opts):
        saved = YahooFantasyCollector().collect_draft(opts['year'], weeks_counted=opts['weeks'])
        self.stdout.write(self.style.SUCCESS(f"Stored {saved} draft picks for {opts['year']}"))
