from django.db import models


class TeamPerformance(models.Model):
    team_name = models.CharField(max_length=100)
    qb_points = models.FloatField()
    wr_points = models.FloatField()
    wr_points_total = models.FloatField()
    rb_points = models.FloatField()
    rb_points_total = models.FloatField()
    te_points = models.FloatField()
    te_points_total = models.FloatField()
    k_points = models.FloatField()
    def_points = models.FloatField()
    total_points = models.FloatField()
    expected_total = models.FloatField()
    difference = models.FloatField()
    points_against = models.FloatField()
    projected_wins = models.CharField(max_length=100, null=True, blank=True)
    actual_wins = models.IntegerField(null=True, blank=True)
    wins_diff = models.IntegerField(null=True, blank=True)
    result = models.CharField(max_length=100, null=True, blank=True)
    week = models.CharField(max_length=100)
    opponent = models.CharField(max_length=100, null=True, blank=True)
    margin = models.FloatField(null=True, blank=True)
    year = models.IntegerField(default=2023)  # Default to 2023 for existing data

    class Meta:
        unique_together = ('team_name', 'week', 'year')

    def __str__(self):
        return f"{self.team_name} - Week {self.week} ({self.year})"

    # def __str__(self):
    #     return self.team_name
