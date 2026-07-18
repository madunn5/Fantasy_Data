from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("draftgame", "0013_seed_seasons"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="punishmentranking",
            name="punishment_index",
        ),
        migrations.AlterUniqueTogether(
            name="punishmentvote",
            unique_together={("user", "season")},
        ),
        migrations.AlterUniqueTogether(
            name="punishmentranking",
            unique_together={("vote", "rank"), ("vote", "punishment")},
        ),
    ]
