from django.db import migrations


def assign_to_2025(apps, schema_editor):
    Season = apps.get_model("draftgame", "Season")
    WheelResult = apps.get_model("draftgame", "WheelResult")
    BucketSession = apps.get_model("draftgame", "BucketSession")
    BucketWeekResult = apps.get_model("draftgame", "BucketWeekResult")

    season_2025 = Season.objects.filter(year=2025).first()
    if season_2025 is None:
        return

    WheelResult.objects.filter(season__isnull=True).update(season=season_2025)
    BucketSession.objects.filter(season__isnull=True).update(season=season_2025)
    BucketWeekResult.objects.filter(season__isnull=True).update(season=season_2025)


def clear_season(apps, schema_editor):
    WheelResult = apps.get_model("draftgame", "WheelResult")
    BucketSession = apps.get_model("draftgame", "BucketSession")
    BucketWeekResult = apps.get_model("draftgame", "BucketWeekResult")
    WheelResult.objects.update(season=None)
    BucketSession.objects.update(season=None)
    BucketWeekResult.objects.update(season=None)


class Migration(migrations.Migration):

    dependencies = [
        ("draftgame", "0016_season_on_wheel_and_bucket"),
    ]

    operations = [
        migrations.RunPython(assign_to_2025, clear_season),
    ]
