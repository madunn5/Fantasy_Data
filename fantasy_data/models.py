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


class ProjectedPoint(models.Model):
    league_key = models.CharField(max_length=32, db_index=True)
    week = models.PositiveIntegerField(db_index=True)
    player_id = models.PositiveIntegerField(db_index=True)
    player_name = models.CharField(max_length=128, blank=True)
    position = models.CharField(max_length=16, blank=True)
    team_key = models.CharField(max_length=32, blank=True)
    projected_total = models.DecimalField(max_digits=7, decimal_places=2, null=True, blank=True)
    projected_raw = models.JSONField(default=dict, blank=True)
    source = models.CharField(max_length=16, default="yahoo")
    pulled_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("league_key", "player_id", "week", "source")
        indexes = [
            models.Index(fields=["league_key", "week"]),
            models.Index(fields=["player_id", "week"]),
        ]

    def __str__(self):
        return f"{self.league_key} | W{self.week} | {self.player_name} ({self.player_id}) -> {self.projected_total}"


class TeamOwnerMapping(models.Model):
    team_name = models.CharField(max_length=100, db_index=True)
    owner_name = models.CharField(max_length=100)
    year = models.IntegerField(db_index=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ('team_name', 'year')
        indexes = [
            models.Index(fields=['team_name', 'year']),
            models.Index(fields=['owner_name', 'year']),
        ]
    
    def __str__(self):
        return f"{self.team_name} -> {self.owner_name} ({self.year})"
    
    @classmethod
    def get_owner_name(cls, team_name, year):
        """Get owner name for a team, fallback to team name if not found"""
        try:
            mapping = cls.objects.get(team_name=team_name, year=year, is_active=True)
            return mapping.owner_name
        except cls.DoesNotExist:
            return team_name  # Fallback to original team name
