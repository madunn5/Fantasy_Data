"""Local test settings: run the site against a throwaway copy of the DB.

Used to test things like the punishment wheel without touching the real
local data. All writes go to db_wheeltest.sqlite3, a copy of db.sqlite3.
Re-copy the file any time you want a fresh start:

    cp db.sqlite3 db_wheeltest.sqlite3
    /Users/Matt/venv/bin/python manage.py runserver 8011 --settings=fantasy_app.wheeltest_settings
"""
from .settings import *  # noqa: F401,F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db_wheeltest.sqlite3",  # noqa: F405
    }
}
