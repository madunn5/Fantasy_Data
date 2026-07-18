from django.db import migrations, models


def set_2025_loser(apps, schema_editor):
    Season = apps.get_model("draftgame", "Season")
    Season.objects.filter(year=2025).update(loser_name="Tobin")


def unset_2025_loser(apps, schema_editor):
    Season = apps.get_model("draftgame", "Season")
    Season.objects.filter(year=2025).update(loser_name="")


class Migration(migrations.Migration):

    dependencies = [
        ("draftgame", "0014_finalize_voting_schema"),
    ]

    operations = [
        migrations.AddField(
            model_name="season",
            name="loser_name",
            field=models.CharField(
                blank=True,
                max_length=100,
                help_text="The last-place finisher who had to carry out the winning punishment.",
            ),
        ),
        migrations.RunPython(set_2025_loser, unset_2025_loser),
    ]
