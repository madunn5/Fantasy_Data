# Fantasy_Data — project guide for Claude

Django fantasy-football analytics app (Heroku app `dunn-right-fantasy`,
GitHub `madunn5/Fantasy_Data`).

## Workflow preferences
- **Commit directly to `main`** in small, logical commits. Do **not** create
  feature branches unless I explicitly ask.
- Show diffs / explain changes as you go; commit and push when I ask.

## Environment
- **Interpreter:** `/Users/Matt/venv/bin/python` (has Django 5 + pandas / numpy /
  plotly + the Yahoo stack). There is no in-tree virtualenv.
- **Run the app:** `/Users/Matt/venv/bin/python manage.py runserver`
  (local DB is `db.sqlite3`, already seeded with real production data).
- **Tests:** `/Users/Matt/venv/bin/python manage.py test fantasy_data`
- **Django check:** `/Users/Matt/venv/bin/python manage.py check`

## Layout
- `fantasy_data/views.py` — page views (year-scoped via `year_nav.get_selected_year`).
- `fantasy_data/predictions.py` — score-distribution win model + power ratings.
- `fantasy_data/analytics.py` — standings, luck report, bench report, playoff sim.
- `fantasy_data/year_nav.py` — global season picker (`?year=` → session → latest).
- `fantasy_data/yahoo_collector.py` — pulls weekly data from the Yahoo API.
- Templates in `fantasy_data/templates/fantasy_data/`, base in `fantasy_app/templates/base.html`.

## Data
- `TeamPerformance` (team-week stats; full 2023–2025), `Player` / `PlayerRoster` /
  `PlayerPerformance` (2025 only), `TeamOwnerMapping`. `team_name` is the person's name.
- Re-seed local DB from the Heroku snapshot:
  `manage.py import_heroku_backup ~/Documents/Fantasy_Website/heroku_backups/dunn-right-fantasy.dump`

## Deploy / config
- `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS` are env-driven
  (`DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`).

## Known issue
- `yahoo_collector.py` sets `Projected Result` = actual `Result`, so the Luck
  Report's "vs Projection" column would read 0 for live-collected data. The
  primary "scoring luck" metric (all-play) is unaffected.
