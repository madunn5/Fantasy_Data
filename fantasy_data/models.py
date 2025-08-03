from django.db import models


class TeamPerformance(models.Model):
    team_name = models.CharField(max_length=100, db_index=True)
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
    week = models.CharField(max_length=100, db_index=True)
    opponent = models.CharField(max_length=100, null=True, blank=True)
    margin = models.FloatField(null=True, blank=True)
    year = models.IntegerField(default=2023, db_index=True)  # Default to 2023 for existing data

    class Meta:
        unique_together = ('team_name', 'week', 'year')
        indexes = [
            models.Index(fields=['team_name', 'year']),
            models.Index(fields=['year', 'week']),
        ]

    def __str__(self):
        return f"{self.team_name} - Week {self.week} ({self.year})"

    # def __str__(self):
    #     return self.team_name


class Player(models.Model):
    yahoo_player_id = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=100)
    position = models.CharField(max_length=10)
    nfl_team = models.CharField(max_length=50, null=True, blank=True)
    
    def __str__(self):
        return f"{self.name} ({self.position})"


class PlayerRoster(models.Model):
    player = models.ForeignKey(Player, on_delete=models.CASCADE)
    fantasy_team = models.CharField(max_length=100, db_index=True)
    week = models.CharField(max_length=100, db_index=True)
    year = models.IntegerField(db_index=True)
    roster_status = models.CharField(max_length=20, choices=[
        ('STARTER', 'Starter'),
        ('BENCH', 'Bench'),
        ('WAIVER', 'Waiver Wire'),
        ('FREE_AGENT', 'Free Agent')
    ])
    
    class Meta:
        unique_together = ('player', 'fantasy_team', 'week', 'year')
        indexes = [
            models.Index(fields=['fantasy_team', 'year']),
            models.Index(fields=['year', 'week']),
        ]


class PlayerPerformance(models.Model):
    player = models.ForeignKey(Player, on_delete=models.CASCADE)
    fantasy_team = models.CharField(max_length=100)
    week = models.CharField(max_length=100)
    year = models.IntegerField()
    points_scored = models.FloatField()
    was_started = models.BooleanField(default=False)
    
    class Meta:
        unique_together = ('player', 'week', 'year')


class PlayerTransaction(models.Model):
    player = models.ForeignKey(Player, on_delete=models.CASCADE)
    from_team = models.CharField(max_length=100, null=True, blank=True)
    to_team = models.CharField(max_length=100, null=True, blank=True)
    transaction_type = models.CharField(max_length=20, choices=[
        ('TRADE', 'Trade'),
        ('PICKUP', 'Waiver/FA Pickup'),
        ('DROP', 'Drop to Waiver'),
        ('DRAFT', 'Draft')
    ])
    week = models.CharField(max_length=100)
    year = models.IntegerField()
    transaction_date = models.DateTimeField(auto_now_add=True)
