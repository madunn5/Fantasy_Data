from django.core.management.base import BaseCommand

from fantasy_data.models import Player, PlayerPerformance, TeamPerformance
from fantasy_data.predictions import week_number
from fantasy_data.projections import collect_projections


class Command(BaseCommand):
    help = "Pull weekly player projections from Sleeper for rostered players."

    def add_arguments(self, parser):
        parser.add_argument('--year', type=int, required=True)
        parser.add_argument('--week', type=int, help='One week (default: every week collected so far)')

    def handle(self, *args, **opts):
        year = opts['year']
        if opts['week']:
            weeks = [opts['week']]
        else:
            weeks = sorted({week_number(w) for w in TeamPerformance.objects.filter(year=year)
                            .values_list('week', flat=True)} - {0})
        players = Player.objects.filter(
            id__in=PlayerPerformance.objects.filter(year=year).values('player_id')).distinct()
        result = collect_projections(year, weeks, players=players)
        self.stdout.write(self.style.SUCCESS(
            f"Stored {result['stored']} projections for {year}, weeks {weeks}"))
        if result['unmatched']:
            self.stdout.write(f"No Sleeper match for: {', '.join(result['unmatched'])}")
