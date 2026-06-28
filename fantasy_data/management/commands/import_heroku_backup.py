"""Import real production data from the dunn-right-fantasy Heroku Postgres backup.

The Heroku Postgres add-on was destroyed on 2026-05-06; the final snapshot lives
at ~/Documents/Fantasy_Website/heroku_backups/dunn-right-fantasy.dump (a custom
pg_dump). This command loads that data into the local database so we can work off
the real season history.

It accepts either the custom .dump (converted internally via `pg_restore`) or an
already-plain .sql file (from `pg_restore -f out.sql <dump>`).

Re-runnable: it clears and reloads the fantasy_data_* tables. It does NOT touch
auth_user / admin (no fantasy_data model references the user table), so your local
logins stay intact.

Usage:
    python manage.py import_heroku_backup ~/Documents/Fantasy_Website/heroku_backups/dunn-right-fantasy.dump
"""
import os
import subprocess
import tempfile

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils.dateparse import parse_datetime

from fantasy_data.models import (
    Player, TeamPerformance, PlayerRoster, PlayerPerformance,
    PlayerTransaction, TeamOwnerMapping, ProjectedPoint,
)


def _unescape(value):
    """Decode a single Postgres COPY field (\\N is NULL)."""
    if value == r"\N":
        return None
    out, i = [], 0
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
    """Yield (table_name, [row_dicts]) for each COPY ... FROM stdin; block."""
    lines = sql_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("COPY ") and "FROM stdin;" in line:
            table = line.split('"."')[1].split('"')[0]
            cols = [c.strip().strip('"')
                    for c in line[line.index("(") + 1:line.index(") FROM stdin;")].split(",")]
            rows = []
            i += 1
            while i < len(lines) and lines[i] != r"\.":
                values = [_unescape(f) for f in lines[i].split("\t")]
                rows.append(dict(zip(cols, values)))
                i += 1
            yield table, rows
        i += 1


def _f(v):  # float or None
    return None if v is None else float(v)


def _i(v):  # int or None
    return None if v is None else int(v)


def _b(v):  # postgres boolean
    return v == "t"


def _dt(v):
    return parse_datetime(v) if v else None


class Command(BaseCommand):
    help = "Import production data from the dunn-right-fantasy Heroku Postgres backup."

    def add_arguments(self, parser):
        parser.add_argument("backup_path", help="Path to the .dump (custom) or .sql (plain) backup")

    def _load_sql_text(self, path):
        if not os.path.exists(path):
            raise CommandError(f"Backup not found: {path}")
        # Sniff: custom pg_dump starts with the magic bytes "PGDMP".
        with open(path, "rb") as fh:
            head = fh.read(5)
        if head == b"PGDMP":
            self.stdout.write("Converting custom dump to SQL via pg_restore...")
            with tempfile.NamedTemporaryFile(suffix=".sql", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                subprocess.run(["pg_restore", "-f", tmp_path, path], check=True)
                with open(tmp_path, encoding="utf-8", errors="replace") as fh:
                    return fh.read()
            except FileNotFoundError:
                raise CommandError("pg_restore not found. Install postgres or pass a plain .sql file.")
            except subprocess.CalledProcessError as exc:
                raise CommandError(f"pg_restore failed: {exc}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()

    @transaction.atomic
    def handle(self, *args, **options):
        blocks = {tbl: rows for tbl, rows in parse_copy_blocks(self._load_sql_text(options["backup_path"]))}

        def rows(name):
            return blocks.get(name, [])

        # Clear current data (FK-safe order) then reload from the backup.
        for model in (PlayerPerformance, PlayerRoster, PlayerTransaction,
                      ProjectedPoint, TeamPerformance, TeamOwnerMapping, Player):
            model.objects.all().delete()

        # --- Players (PKs preserved so roster/performance FKs line up) -------
        Player.objects.bulk_create([
            Player(id=_i(r["id"]), yahoo_player_id=r["yahoo_player_id"], name=r["name"],
                   position=r["position"], nfl_team=r.get("nfl_team"))
            for r in rows("fantasy_data_player")
        ])

        # --- Team performance ----------------------------------------------
        TeamPerformance.objects.bulk_create([
            TeamPerformance(
                id=_i(r["id"]), team_name=r["team_name"],
                qb_points=_f(r["qb_points"]), wr_points=_f(r["wr_points"]),
                wr_points_total=_f(r["wr_points_total"]), rb_points=_f(r["rb_points"]),
                rb_points_total=_f(r["rb_points_total"]), te_points=_f(r["te_points"]),
                te_points_total=_f(r["te_points_total"]), k_points=_f(r["k_points"]),
                def_points=_f(r["def_points"]), total_points=_f(r["total_points"]),
                expected_total=_f(r["expected_total"]), difference=_f(r["difference"]),
                points_against=_f(r["points_against"]), projected_wins=r.get("projected_wins"),
                actual_wins=_i(r.get("actual_wins")), wins_diff=_i(r.get("wins_diff")),
                result=r.get("result"), week=r["week"], opponent=r.get("opponent"),
                margin=_f(r.get("margin")), year=_i(r["year"]),
            )
            for r in rows("fantasy_data_teamperformance")
        ])

        # --- Player rosters -------------------------------------------------
        PlayerRoster.objects.bulk_create([
            PlayerRoster(id=_i(r["id"]), player_id=_i(r["player_id"]),
                         fantasy_team=r["fantasy_team"], week=r["week"],
                         year=_i(r["year"]), roster_status=r["roster_status"])
            for r in rows("fantasy_data_playerroster")
        ])

        # --- Player performances -------------------------------------------
        PlayerPerformance.objects.bulk_create([
            PlayerPerformance(id=_i(r["id"]), player_id=_i(r["player_id"]),
                              fantasy_team=r["fantasy_team"], week=r["week"],
                              year=_i(r["year"]), points_scored=_f(r["points_scored"]),
                              was_started=_b(r["was_started"]))
            for r in rows("fantasy_data_playerperformance")
        ])

        # --- Player transactions -------------------------------------------
        for r in rows("fantasy_data_playertransaction"):
            t = PlayerTransaction.objects.create(
                id=_i(r["id"]), player_id=_i(r["player_id"]),
                from_team=r.get("from_team"), to_team=r.get("to_team"),
                transaction_type=r["transaction_type"], week=r["week"], year=_i(r["year"]),
            )
            if r.get("transaction_date"):
                PlayerTransaction.objects.filter(pk=t.pk).update(transaction_date=_dt(r["transaction_date"]))

        # --- Team-owner mappings (preserve created/updated timestamps) ------
        for r in rows("fantasy_data_teamownermapping"):
            m = TeamOwnerMapping.objects.create(
                id=_i(r["id"]), team_name=r["team_name"], owner_name=r["owner_name"],
                year=_i(r["year"]), is_active=_b(r["is_active"]),
            )
            TeamOwnerMapping.objects.filter(pk=m.pk).update(
                created_at=_dt(r.get("created_at")) or m.created_at,
                updated_at=_dt(r.get("updated_at")) or m.updated_at,
            )

        # --- Projected points (carry over if present) ----------------------
        ProjectedPoint.objects.bulk_create([
            ProjectedPoint(
                id=_i(r["id"]), league_key=r["league_key"], week=_i(r["week"]),
                player_id=_i(r["player_id"]), player_name=r.get("player_name") or "",
                position=r.get("position") or "", team_key=r.get("team_key") or "",
                projected_total=r.get("projected_total"),
                source=r.get("source") or "yahoo",
            )
            for r in rows("fantasy_data_projectedpoint")
        ])

        self.stdout.write(self.style.SUCCESS(
            "Heroku backup import complete:\n"
            f"  players:             {Player.objects.count()}\n"
            f"  team performances:   {TeamPerformance.objects.count()}\n"
            f"  player rosters:      {PlayerRoster.objects.count()}\n"
            f"  player performances: {PlayerPerformance.objects.count()}\n"
            f"  transactions:        {PlayerTransaction.objects.count()}\n"
            f"  team-owner mappings: {TeamOwnerMapping.objects.count()}\n"
            f"  projected points:    {ProjectedPoint.objects.count()}"
        ))
