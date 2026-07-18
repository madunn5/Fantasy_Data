"""
Import the real 2025 production data from a Heroku Postgres backup (custom dump
converted to plain SQL via `pg_restore -f out.sql backup.dump`).

Loads the legacy data into the new season-based schema:
  - auth_user rows (preserving password hashes so members can still log in)
  - punishment votes + rankings -> attached to the 2025 Season, with the old
    integer punishment_index mapped onto the 2025 seed Punishment rows
  - wheel results and the full Bucket of Death history (schema unchanged)

Re-runnable: it clears the dynamic tables (votes, rankings, wheel + bucket
results) and reloads them. It does NOT touch Season rows or the seed Punishment
pool created by the migrations.

Usage:
    python manage.py import_2025_backup /path/to/prod.sql
"""
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.contrib.auth.models import User
from django.utils.dateparse import parse_datetime
import json

from draftgame.models import (
    Season, Punishment, PunishmentVote, PunishmentRanking,
    WheelResult, BucketSession, BucketWeekResult, BucketTeamResult,
)


def _unescape(value):
    """Decode a single Postgres COPY field (\\N is NULL)."""
    if value == r"\N":
        return None
    out = []
    i = 0
    while i < len(value):
        c = value[i]
        if c == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            out.append({"t": "\t", "n": "\n", "r": "\r", "\\": "\\"}.get(nxt, nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def parse_copy_blocks(sql_text):
    """Yield (table_name, [column_names], [row_dicts]) for each COPY block."""
    lines = sql_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("COPY ") and "FROM stdin;" in line:
            # COPY "public"."table" ("col1", "col2", ...) FROM stdin;
            header = line
            table = header.split('"."')[1].split('"')[0]
            cols_part = header[header.index("(") + 1:header.index(") FROM stdin;")]
            columns = [c.strip().strip('"') for c in cols_part.split(",")]
            rows = []
            i += 1
            while i < len(lines) and lines[i] != r"\.":
                raw = lines[i]
                fields = raw.split("\t")
                values = [_unescape(f) for f in fields]
                rows.append(dict(zip(columns, values)))
                i += 1
            yield table, columns, rows
        i += 1


def _to_bool(v):
    return v == "t"


def _to_dt(v):
    return parse_datetime(v) if v else None


class Command(BaseCommand):
    help = "Import real 2025 production data from a pg_restore SQL dump."

    def add_arguments(self, parser):
        parser.add_argument("sql_path", help="Path to the SQL file produced by pg_restore -f")

    @transaction.atomic
    def handle(self, *args, **options):
        path = options["sql_path"]
        try:
            with open(path, encoding="utf-8") as fh:
                sql_text = fh.read()
        except OSError as exc:
            raise CommandError(f"Could not read {path}: {exc}")

        blocks = {table: rows for table, cols, rows in parse_copy_blocks(sql_text)}

        season_2025 = Season.objects.filter(year=2025).first()
        if season_2025 is None:
            raise CommandError("2025 Season not found. Run migrations first.")

        # The 2025 seed pool, ordered to match the legacy punishment_index.
        seed_2025 = list(season_2025.punishments.order_by("id"))
        if not seed_2025:
            raise CommandError("2025 Season has no seed punishments. Run migrations first.")

        # --- Users (preserve password hashes / staff flags) -----------------
        user_id_map = {}  # legacy id -> User instance
        for row in blocks.get("auth_user", []):
            user, _ = User.objects.get_or_create(username=row["username"])
            user.password = row["password"] or user.password
            user.email = row["email"] or ""
            user.is_staff = _to_bool(row["is_staff"])
            user.is_superuser = _to_bool(row["is_superuser"])
            user.is_active = _to_bool(row["is_active"])
            user.first_name = row.get("first_name") or ""
            user.last_name = row.get("last_name") or ""
            if row.get("date_joined"):
                user.date_joined = _to_dt(row["date_joined"])
            if row.get("last_login"):
                user.last_login = _to_dt(row["last_login"])
            user.save()
            user_id_map[row["id"]] = user
        self.stdout.write(f"Users imported/updated: {len(user_id_map)}")

        # --- Clear dynamic data we are about to reload ----------------------
        PunishmentRanking.objects.all().delete()
        PunishmentVote.objects.all().delete()
        WheelResult.objects.all().delete()
        BucketTeamResult.objects.all().delete()
        BucketWeekResult.objects.all().delete()
        BucketSession.objects.all().delete()

        # --- Votes + rankings (attach to 2025) ------------------------------
        vote_id_map = {}  # legacy vote id -> PunishmentVote
        for row in blocks.get("draftgame_punishmentvote", []):
            user = user_id_map.get(row["user_id"])
            if user is None:
                continue
            vote = PunishmentVote.objects.create(user=user, season=season_2025)
            PunishmentVote.objects.filter(pk=vote.pk).update(
                created_at=_to_dt(row.get("created_at")) or vote.created_at,
                updated_at=_to_dt(row.get("updated_at")) or vote.updated_at,
            )
            vote_id_map[row["id"]] = vote

        rankings_made = 0
        for row in blocks.get("draftgame_punishmentranking", []):
            vote = vote_id_map.get(row["vote_id"])
            if vote is None:
                continue
            idx = int(row["punishment_index"])
            if not (0 <= idx < len(seed_2025)):
                continue
            PunishmentRanking.objects.create(
                vote=vote, punishment=seed_2025[idx], rank=int(row["rank"])
            )
            rankings_made += 1
        self.stdout.write(f"Votes imported: {len(vote_id_map)}, rankings: {rankings_made}")

        # --- Wheel results --------------------------------------------------
        for row in blocks.get("draftgame_wheelresult", []):
            wr = WheelResult.objects.create(name=row["name"], punishment=row["punishment"])
            WheelResult.objects.filter(pk=wr.pk).update(
                created_at=_to_dt(row.get("created_at")) or wr.created_at
            )
        self.stdout.write(f"Wheel results imported: {len(blocks.get('draftgame_wheelresult', []))}")

        # --- Bucket of Death history ----------------------------------------
        for row in blocks.get("draftgame_bucketsession", []):
            bs = BucketSession.objects.create(
                session_key=row["session_key"],
                punishment=row.get("punishment") or "",
                kept_teams=json.loads(row["kept_teams"]) if row.get("kept_teams") else [],
                must_keep_next=_to_bool(row.get("must_keep_next")),
                week_number=int(row.get("week_number") or 1),
                is_test_mode=_to_bool(row.get("is_test_mode")),
                current_player=row.get("current_player") or "",
                selected_team=row.get("selected_team") or "",
            )
            BucketSession.objects.filter(pk=bs.pk).update(
                created_at=_to_dt(row.get("created_at")) or bs.created_at,
                updated_at=_to_dt(row.get("updated_at")) or bs.updated_at,
            )

        week_id_map = {}  # legacy id -> BucketWeekResult
        for row in blocks.get("draftgame_bucketweekresult", []):
            wk = BucketWeekResult.objects.create(
                week_number=int(row["week_number"]),
                punishment=row.get("punishment") or "",
                kept_teams=json.loads(row["kept_teams"]) if row.get("kept_teams") else [],
            )
            BucketWeekResult.objects.filter(pk=wk.pk).update(
                created_at=_to_dt(row.get("created_at")) or wk.created_at
            )
            week_id_map[row["id"]] = wk

        team_results = 0
        for row in blocks.get("draftgame_bucketteamresult", []):
            week = week_id_map.get(row["week_result_id"])
            if week is None:
                continue
            is_win = None if row.get("is_win") is None else _to_bool(row["is_win"])
            tr = BucketTeamResult.objects.create(
                week_result=week,
                team_name=row["team_name"],
                player_name=row["player_name"],
                is_win=is_win,
            )
            BucketTeamResult.objects.filter(pk=tr.pk).update(
                created_at=_to_dt(row.get("created_at")) or tr.created_at
            )
            team_results += 1
        self.stdout.write(
            f"Bucket: {len(blocks.get('draftgame_bucketsession', []))} sessions, "
            f"{len(week_id_map)} weeks, {team_results} team results"
        )

        self.stdout.write(self.style.SUCCESS("2025 backup import complete."))
