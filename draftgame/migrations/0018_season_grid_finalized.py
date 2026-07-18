from django.db import migrations, models


def finalize_past_seasons(apps, schema_editor):
    # Seasons that already have a recorded winner are done — lock their grids.
    Season = apps.get_model("draftgame", "Season")
    Punishment = apps.get_model("draftgame", "Punishment")
    won_season_ids = Punishment.objects.filter(is_winner=True).values_list("season_id", flat=True)
    Season.objects.filter(id__in=list(won_season_ids)).update(grid_finalized=True)


def unfinalize(apps, schema_editor):
    Season = apps.get_model("draftgame", "Season")
    Season.objects.update(grid_finalized=False)


class Migration(migrations.Migration):

    dependencies = [
        ("draftgame", "0017_backfill_wheel_bucket_season"),
    ]

    operations = [
        migrations.AddField(
            model_name="season",
            name="grid_finalized",
            field=models.BooleanField(
                default=False,
                help_text="When set, the wheel is locked and grid assignments can no longer be changed.",
            ),
        ),
        migrations.RunPython(finalize_past_seasons, unfinalize),
    ]
