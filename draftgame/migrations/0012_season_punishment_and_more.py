from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("draftgame", "0011_bucketteamresult"),
    ]

    operations = [
        migrations.CreateModel(
            name="Season",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("year", models.IntegerField(unique=True)),
                ("is_active", models.BooleanField(default=False, help_text="The current season. Only one should be active at a time.")),
                ("submissions_open", models.BooleanField(default=False, help_text="Allow authenticated users to submit new punishments.")),
                ("voting_open", models.BooleanField(default=False, help_text="Allow authenticated users to rank punishments.")),
                ("ranking_size", models.IntegerField(default=12, help_text="How many punishments each voter ranks (their top N).")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={"ordering": ["-year"]},
        ),
        migrations.CreateModel(
            name="Punishment",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("text", models.TextField()),
                ("is_seed", models.BooleanField(default=False, help_text="Carried over from a prior year; does not count against a user's submission limit.")),
                ("is_winner", models.BooleanField(default=False, help_text="The winning punishment for this season.")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("season", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="punishments", to="draftgame.season")),
                ("submitted_by", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="submitted_punishments", to=settings.AUTH_USER_MODEL)),
            ],
            options={"ordering": ["season", "id"]},
        ),
        migrations.AddField(
            model_name="punishmentvote",
            name="season",
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name="votes", to="draftgame.season"),
        ),
        migrations.AlterField(
            model_name="punishmentvote",
            name="user",
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="punishment_votes", to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name="punishmentranking",
            name="punishment",
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name="rankings", to="draftgame.punishment"),
        ),
        migrations.AlterField(
            model_name="punishmentranking",
            name="rank",
            field=models.IntegerField(validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(100)]),
        ),
        # Drop the punishment_index-based unique constraint so the column can be removed in 0014.
        migrations.AlterUniqueTogether(
            name="punishmentranking",
            unique_together={("vote", "rank")},
        ),
    ]
