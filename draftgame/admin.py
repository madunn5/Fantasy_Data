from django.contrib import admin
from .models import (
    Season, Punishment, PunishmentVote, PunishmentRanking, WheelResult,
    BucketSession, BucketWeekResult, BucketTeamResult,
)


@admin.register(Season)
class SeasonAdmin(admin.ModelAdmin):
    list_display = ['year', 'is_active', 'submissions_open', 'voting_open', 'grid_finalized', 'bucket_open', 'ranking_size', 'wheel_size', 'loser_name']
    list_editable = ['is_active', 'submissions_open', 'voting_open', 'grid_finalized', 'bucket_open', 'ranking_size', 'wheel_size', 'loser_name']


@admin.register(Punishment)
class PunishmentAdmin(admin.ModelAdmin):
    list_display = ['short_text', 'season', 'submitted_by', 'is_seed', 'is_finalist', 'is_winner', 'created_at']
    list_filter = ['season', 'is_seed', 'is_finalist', 'is_winner']
    list_editable = ['is_finalist', 'is_winner']
    search_fields = ['text']

    @admin.display(description='Punishment')
    def short_text(self, obj):
        return obj.text[:80]


class PunishmentRankingInline(admin.TabularInline):
    model = PunishmentRanking
    extra = 0


@admin.register(PunishmentVote)
class PunishmentVoteAdmin(admin.ModelAdmin):
    list_display = ['user', 'season', 'created_at', 'updated_at']
    list_filter = ['season']
    inlines = [PunishmentRankingInline]


@admin.register(PunishmentRanking)
class PunishmentRankingAdmin(admin.ModelAdmin):
    list_display = ['vote', 'punishment', 'rank']
    list_filter = ['rank', 'vote__season']


@admin.register(WheelResult)
class WheelResultAdmin(admin.ModelAdmin):
    list_display = ['name', 'season', 'short_punishment', 'created_at']
    list_filter = ['season']

    @admin.display(description='Punishment')
    def short_punishment(self, obj):
        return obj.punishment[:60]


class BucketTeamResultInline(admin.TabularInline):
    model = BucketTeamResult
    extra = 0


@admin.register(BucketSession)
class BucketSessionAdmin(admin.ModelAdmin):
    list_display = ['session_key', 'season', 'current_player', 'week_number', 'is_test_mode', 'created_at']
    list_filter = ['season', 'week_number', 'is_test_mode']
    readonly_fields = ['session_key', 'created_at', 'updated_at']


@admin.register(BucketWeekResult)
class BucketWeekResultAdmin(admin.ModelAdmin):
    list_display = ['week_number', 'season', 'punishment', 'created_at']
    list_filter = ['season']
    inlines = [BucketTeamResultInline]


@admin.register(BucketTeamResult)
class BucketTeamResultAdmin(admin.ModelAdmin):
    list_display = ['week_result', 'team_name', 'player_name', 'is_win', 'created_at']
    list_filter = ['is_win', 'week_result__week_number']
