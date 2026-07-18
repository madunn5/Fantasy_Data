from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone


class Season(models.Model):
    year = models.IntegerField(unique=True)
    is_active = models.BooleanField(default=False, help_text="The current season. Only one should be active at a time.")
    submissions_open = models.BooleanField(default=False, help_text="Allow authenticated users to submit new punishments.")
    submissions_close_at = models.DateTimeField(
        null=True, blank=True,
        help_text="If set, submissions automatically close at this time even while submissions_open is on."
    )
    voting_open = models.BooleanField(default=False, help_text="Allow authenticated users to rank punishments.")
    ranking_size = models.IntegerField(default=12, help_text="How many punishments each voter ranks (their top N).")
    wheel_size = models.IntegerField(default=12, help_text="How many top-voted punishments are locked in as the season's wheel/grid set.")
    loser_name = models.CharField(max_length=100, blank=True, help_text="The last-place finisher who had to carry out the winning punishment.")
    grid_finalized = models.BooleanField(default=False, help_text="When set, the wheel is locked and grid assignments can no longer be changed.")
    bucket_open = models.BooleanField(default=False, help_text="When set, Bucket of Death is live and visible to the whole league (staff can always see it).")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-year']

    def __str__(self):
        label = f"{self.year} Season"
        if self.is_active:
            label += " (active)"
        return label

    @classmethod
    def get_active(cls):
        return cls.objects.filter(is_active=True).order_by('-year').first()

    @property
    def submissions_are_open(self):
        """Submissions flag combined with the optional deadline."""
        if not self.submissions_open:
            return False
        if self.submissions_close_at and timezone.now() >= self.submissions_close_at:
            return False
        return True

    @property
    def voting_complete(self):
        """Voting has closed and the grid set is locked in."""
        return not self.voting_open and self.punishments.filter(is_finalist=True).exists()

    def save(self, *args, **kwargs):
        # Enforce a single active season.
        if self.is_active:
            Season.objects.exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)


class Punishment(models.Model):
    season = models.ForeignKey(Season, on_delete=models.CASCADE, related_name='punishments')
    text = models.TextField()
    submitted_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='submitted_punishments'
    )
    is_seed = models.BooleanField(default=False, help_text="Carried over from a prior year; does not count against a user's submission limit.")
    is_duplicate = models.BooleanField(default=False, help_text="Consolidated as a duplicate of an earlier submission. Stays visible on the suggestion list but is excluded from voting.")
    is_finalist = models.BooleanField(default=False, help_text="Locked in (via Finalize voting) as part of the season's wheel/grid set.")
    is_winner = models.BooleanField(default=False, help_text="The winning punishment for this season.")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['season', 'id']

    def __str__(self):
        return self.text[:60]


class PunishmentVote(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='punishment_votes')
    season = models.ForeignKey(Season, on_delete=models.CASCADE, related_name='votes', null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [['user', 'season']]

    def __str__(self):
        return f"{self.user.username} - {self.season}"


class PunishmentRanking(models.Model):
    vote = models.ForeignKey(PunishmentVote, on_delete=models.CASCADE, related_name='rankings')
    punishment = models.ForeignKey(Punishment, on_delete=models.CASCADE, related_name='rankings', null=True)
    rank = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(100)])

    class Meta:
        unique_together = [['vote', 'rank'], ['vote', 'punishment']]


class WheelResult(models.Model):
    season = models.ForeignKey(Season, on_delete=models.CASCADE, related_name='wheel_results', null=True)
    name = models.CharField(max_length=100)
    punishment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)


class BucketSession(models.Model):
    season = models.ForeignKey(Season, on_delete=models.CASCADE, related_name='bucket_sessions', null=True)
    session_key = models.CharField(max_length=40)
    punishment = models.TextField(blank=True)
    kept_teams = models.JSONField(default=list)
    must_keep_next = models.BooleanField(default=False)
    week_number = models.IntegerField(default=1)
    is_test_mode = models.BooleanField(default=False)
    current_player = models.CharField(max_length=100, blank=True)
    selected_team = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [['session_key', 'season']]


class BucketWeekResult(models.Model):
    season = models.ForeignKey(Season, on_delete=models.CASCADE, related_name='bucket_weeks', null=True)
    week_number = models.IntegerField()
    punishment = models.TextField()
    kept_teams = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['season', 'week_number']]


class BucketTeamResult(models.Model):
    week_result = models.ForeignKey(BucketWeekResult, on_delete=models.CASCADE, related_name='team_results')
    team_name = models.CharField(max_length=100)
    player_name = models.CharField(max_length=100)
    is_win = models.BooleanField(null=True, blank=True)  # None=pending, True=win, False=loss
    created_at = models.DateTimeField(auto_now_add=True)
