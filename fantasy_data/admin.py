from django.contrib import admin
from fantasy_data.models import TeamPerformance, Player, PlayerRoster, PlayerPerformance, PlayerTransaction

@admin.register(TeamPerformance)
class TeamPerformanceAdmin(admin.ModelAdmin):
    list_display = ['team_name', 'week', 'year', 'total_points', 'result']
    list_filter = ['year', 'week', 'result']
    search_fields = ['team_name']

@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ['name', 'position', 'nfl_team']
    list_filter = ['position', 'nfl_team']
    search_fields = ['name']

@admin.register(PlayerRoster)
class PlayerRosterAdmin(admin.ModelAdmin):
    list_display = ['player', 'fantasy_team', 'week', 'year', 'roster_status']
    list_filter = ['year', 'week', 'roster_status', 'fantasy_team']
    search_fields = ['player__name', 'fantasy_team']

@admin.register(PlayerPerformance)
class PlayerPerformanceAdmin(admin.ModelAdmin):
    list_display = ['player', 'fantasy_team', 'week', 'year', 'points_scored', 'was_started']
    list_filter = ['year', 'week', 'was_started', 'fantasy_team']
    search_fields = ['player__name', 'fantasy_team']

@admin.register(PlayerTransaction)
class PlayerTransactionAdmin(admin.ModelAdmin):
    list_display = ['player', 'transaction_type', 'from_team', 'to_team', 'week', 'year']
    list_filter = ['year', 'week', 'transaction_type']
    search_fields = ['player__name', 'from_team', 'to_team']
