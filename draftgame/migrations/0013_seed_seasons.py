from django.db import migrations


# The punishments that have existed in the league so far (previously hardcoded in views.py).
# These seed both the 2025 (history) and 2026 (carried-over starting pool) seasons.
SEED_PUNISHMENTS = [
    "Get your nipple pierced",
    "24 hours live on instagram at Waffle House or Dennys where each waffle lowers time spent by 1 hour.",
    "Get a beta fish and keep it alive until the next draft. If the beta fish dies, you must buy two more. If those "
    "both die, then you buy three. If all three die, you get a cat",
    "Post a daily selfie for a month long fitness challenge to all of your followers on Instagram. If you don't have an "
    "instagram you must make one and amass at least 100 followers before the challenge begins.",
    "You must make 10 free throws in a row to complete the punishment. For every missed shot, run one lap around a track — and after every 5 total laps, the required streak drops by one, down to a minimum of 1.",
    "Go to church every sunday.",
    "Bleach your hair",
    "Do the one chip or other spicy food challenge",
    "Play an instrument at nearby college campus until you get $20 in tips",
    "Complete the 151 Pokédex in Pokemon red, blue, or yellow before next year's draft",
    "9-9-9 challenge at a kids baseball game.",
    "You must complete a 500-piece puzzle. You are not allowed to stop or leave the table until you've completed it.",
    "Give everyone in the league $50 each",
    "Play a round of golf another member of the league wearing an outfit of the winner's choosing",
    "You cannot wear sunglasses until the first pick of the 2026 fantasy draft has been made. You have to tell your friends, family, coworkers, etc, and do your best to honor the punishment. Possibly even have your significant other/friends/family lock up or confiscate all of your shades. For each time you mess up and wear sunglasses, you have to Venmo everyone in the league $5 each",
    "Take a picture of yourself while standing under the Gateway Arch in St. Louis, Missouri. Gateway Arch must be visible and photo cannot be edited or AI-generated in any way. Photo must be posted in the league discord before the first pick of the 2026 fantasy draft is made",
    "Must run a 5K, wearing an outfit provided by (and purchased by) the league",
    "Brazilian wax, must record reaction live.",
    "Take the SAT",
    "Print and send out 100 mailers to your entire neighborhood with a picture of you, your name, and your fantasy record announcing your last place.",
    "Must run a mile a day until the season starts again - track via strava or Nike Run",
    "Into the spider verse - must watch all spider man (Toby McGuire, Andrew Garfield, tom Holland, and animated movies) in one sitting",
    "Go to a public library (college or otherwise) and get someone to read you a bed time story",
    "Build a raft, by hand (no pre-purchased kayaks or floating devices) and cross Tempe Town Lake (or other body of water)",
]


def seed_seasons(apps, schema_editor):
    Season = apps.get_model("draftgame", "Season")
    Punishment = apps.get_model("draftgame", "Punishment")
    PunishmentVote = apps.get_model("draftgame", "PunishmentVote")
    PunishmentRanking = apps.get_model("draftgame", "PunishmentRanking")

    season_2025, _ = Season.objects.get_or_create(
        year=2025,
        defaults={"is_active": False, "submissions_open": False, "voting_open": False, "ranking_size": 12},
    )
    season_2026, _ = Season.objects.get_or_create(
        year=2026,
        defaults={"is_active": True, "submissions_open": True, "voting_open": True, "ranking_size": 12},
    )

    # Seed 2025 (history). Keep the ordered list to remap legacy rankings by index.
    p2025 = []
    if not season_2025.punishments.exists():
        for text in SEED_PUNISHMENTS:
            p = Punishment.objects.create(
                season=season_2025,
                text=text,
                is_seed=True,
                is_winner=("Gateway Arch" in text),  # St. Louis Arch won 2025
            )
            p2025.append(p)
    else:
        p2025 = list(season_2025.punishments.order_by("id"))

    # Carry the same pool over to 2026 as the starting set.
    if not season_2026.punishments.exists():
        for text in SEED_PUNISHMENTS:
            Punishment.objects.create(season=season_2026, text=text, is_seed=True)

    # Attach any pre-existing votes/rankings to 2025 and remap index -> punishment row.
    PunishmentVote.objects.filter(season__isnull=True).update(season=season_2025)
    for ranking in PunishmentRanking.objects.filter(punishment__isnull=True):
        idx = getattr(ranking, "punishment_index", None)
        if idx is not None and 0 <= idx < len(p2025):
            ranking.punishment = p2025[idx]
            ranking.save(update_fields=["punishment"])


def unseed_seasons(apps, schema_editor):
    Season = apps.get_model("draftgame", "Season")
    Season.objects.filter(year__in=[2025, 2026]).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("draftgame", "0012_season_punishment_and_more"),
    ]

    operations = [
        migrations.RunPython(seed_seasons, unseed_seasons),
    ]
