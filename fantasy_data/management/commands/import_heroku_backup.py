"""Mirror production data from a dunn-right-fantasy Heroku Postgres backup into the local DB.

Grab a fresh backup first:
    heroku pg:backups:capture -a dunn-right-fantasy
    heroku pg:backups:download -a dunn-right-fantasy -o ~/Documents/Fantasy_Website/heroku_backups/dunn-right-fantasy-YYYY-MM-DD.dump

It accepts either the custom .dump (converted internally via `pg_restore`) or an
already-plain .sql file (from `pg_restore -f out.sql <dump>`).

Every fantasy_data_* and draftgame_* table found in the backup is cleared and
reloaded row for row (primary keys preserved). Tables missing from the backup are
left alone. Auth / sessions / admin are NOT touched, so local logins stay intact.
Prod and local give the same person different user ids, so foreign keys to the
user table are matched by username; a prod user with no local account gets one
created (no usable password).

Everything runs in one transaction and finishes with a foreign-key check, so a
bad backup leaves the local DB as it was. Local DB only: it refuses to run
against anything but SQLite.

Usage:
    python manage.py import_heroku_backup ~/Documents/Fantasy_Website/heroku_backups/dunn-right-fantasy-YYYY-MM-DD.dump
"""
import json
import os
import subprocess
import tempfile

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, models, transaction

MIRRORED_APPS = ("fantasy_data", "draftgame")


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
            table = line[len("COPY "):line.index(" (")].split(".")[-1].strip('"')
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


def _is_user_fk(field):
    return field.is_relation and field.many_to_one and \
        field.related_model._meta.label == settings.AUTH_USER_MODEL


class Command(BaseCommand):
    help = "Mirror fantasy_data and draftgame tables from a dunn-right-fantasy Heroku Postgres backup."

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

    def _user_map(self, prod_users):
        """prod user id -> local user id, matched on username (missing users are created)."""
        User = get_user_model()
        local_ids = dict(User.objects.values_list("username", "id"))
        mapping = {}
        for r in prod_users:
            if r["username"] not in local_ids:
                user = User(username=r["username"], email=r.get("email") or "")
                user.set_unusable_password()
                user.save()
                local_ids[user.username] = user.id
                self.stdout.write(f"  created local user {user.username!r}")
            mapping[r["id"]] = local_ids[r["username"]]
        return mapping

    def _convert(self, field, raw, user_map):
        """Turn one COPY text value into what the local DB column expects."""
        if raw is None:
            return None
        if _is_user_fk(field):
            return user_map[raw]
        if isinstance(field, models.JSONField):
            value = json.loads(raw)
        elif field.is_relation:
            value = field.target_field.to_python(raw)
        else:
            value = field.to_python(raw)
        return field.get_db_prep_save(value, connection)

    def handle(self, *args, **options):
        if connection.vendor != "sqlite":
            raise CommandError("import_heroku_backup only loads into the local SQLite DB.")

        blocks = dict(parse_copy_blocks(self._load_sql_text(options["backup_path"])))
        qn = connection.ops.quote_name

        with transaction.atomic():
            user_map = self._user_map(blocks.get("auth_user", []))
            loaded, skipped = [], []
            for app_label in MIRRORED_APPS:
                for model in apps.get_app_config(app_label).get_models(include_auto_created=True):
                    table = model._meta.db_table
                    if table not in blocks:
                        skipped.append(table)
                        continue
                    rows = blocks[table]
                    fields = {f.column: f for f in model._meta.concrete_fields}
                    if rows and set(rows[0]) != set(fields):
                        raise CommandError(
                            f"{table}: backup columns don't match the local model "
                            f"(backup only: {sorted(set(rows[0]) - set(fields))}, "
                            f"local only: {sorted(set(fields) - set(rows[0]))}). "
                            "Run migrate locally or take a newer backup."
                        )
                    cols = list(fields)
                    values = [[self._convert(fields[c], r[c], user_map) for c in cols] for r in rows]
                    with connection.cursor() as cursor:
                        cursor.execute(f"DELETE FROM {qn(table)}")
                        if values:
                            cursor.executemany(
                                f"INSERT INTO {qn(table)} ({', '.join(qn(c) for c in cols)}) "
                                f"VALUES ({', '.join(['%s'] * len(cols))})",
                                values,
                            )
                    loaded.append((table, len(values)))
            # FKs are deferred on SQLite, so check them before the transaction commits.
            connection.check_constraints(table_names=[t for t, _ in loaded])

        width = max((len(t) for t, _ in loaded), default=0)
        self.stdout.write(self.style.SUCCESS(
            "Heroku backup import complete:\n" +
            "\n".join(f"  {t:<{width}}  {n}" for t, n in loaded)
        ))
        if skipped:
            self.stdout.write(f"Not in backup, left as is: {', '.join(skipped)}")
