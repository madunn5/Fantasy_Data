# Fantasy_Data — project guide for Claude

Combined Django fantasy-football site (GitHub `madunn5/Fantasy_Data`): league
analytics (`fantasy_data`) plus punishments / Bucket of Death (`draftgame`,
merged in from the old fantasy_project repo July 2026).

## URL structure
- `/` — landing page (`fantasy_app/templates/landing.html`), two paths.
- `/data/...` — analytics site (`fantasy_data.urls`, un-namespaced names).
- `/punishments/...` — punishment site (`draftgame.urls`, namespace `draftgame:`).
- `/accounts/...` — shared auth: allauth Google sign-in + username/password
  (`login`/`logout`/`register` are global URL names; draftgame's
  `CustomLoginView` + `logout_view`/`register` are mounted here).

## Workflow preferences
- **Commit directly to `main`** in small, logical commits. Do **not** create
  feature branches unless I explicitly ask.
- Show diffs / explain changes as you go; commit and push when I ask.

## Environment
- **Interpreter:** `/Users/Matt/venv/bin/python` (has Django 5 + pandas / numpy /
  plotly + the Yahoo stack). There is no in-tree virtualenv.
- **Run the app:** `/Users/Matt/venv/bin/python manage.py runserver`
  (local DB is `db.sqlite3`, already seeded with real production data).
- **Tests:** `/Users/Matt/venv/bin/python manage.py test fantasy_data draftgame`
- **Django check:** `/Users/Matt/venv/bin/python manage.py check`

## Layout
- `fantasy_data/views.py` — page views (year-scoped via `year_nav.get_selected_year`).
- `fantasy_data/predictions.py` — score-distribution win model + power ratings.
- `fantasy_data/analytics.py` — standings, luck report, bench report, playoff sim.
- `fantasy_data/year_nav.py` — global season picker (`?year=` → session → latest).
- `fantasy_data/yahoo_collector.py` — pulls weekly data from the Yahoo API.
- Templates in `fantasy_data/templates/fantasy_data/`, base in `fantasy_app/templates/base.html`.
- `draftgame/` — punishment voting, wheel, draft lottery, Bucket of Death.
  Its templates live in `draftgame/templates/` (base: `draftgame/base.html`);
  season picker via `draftgame.context_processors.selected_season`.
- Dev-server note: Django caches templates even with DEBUG on — restart
  `runserver` after editing templates if changes don't show.

## Design ("Midnight Broadcast", July 2026)
- One dark shell for both halves; volt (#e8ff47) accents the data side, red
  (#e03f3f) the punishments side. Volt never appears on punishment pages
  (exception: the wheel pointer); no gradients on buttons.
- Tokens live in three places (keep in sync): `fantasy_data/static/fantasy_data/theme.css`
  (:root, also the Bootstrap/DataTables overrides), `draftgame/templates/draftgame/base.html`
  (:root), `fantasy_app/templates/landing.html` (:root). Charts:
  `fantasy_data/plotly_theme.py` registers the "dunn" default template + fixed
  per-owner colorway.
- Type: Barlow Condensed 600 for display/headings (uppercase), Barlow for body.
- After editing theme.css run `manage.py collectstatic` (manifest storage;
  tests fail on a stale manifest).
- User-facing copy anywhere on the site: no em dashes, and keep it plain and
  conversational — short sentences, no "Higher = X" shorthand, nothing that
  reads like AI wrote it. Explainers should just say what the table shows and
  how to read it.

## Data
- `TeamPerformance` (team-week stats; full 2023–2025), `Player` / `PlayerRoster` /
  `PlayerPerformance` (2025 only), `TeamOwnerMapping`. `team_name` is the person's name.
- Re-seed local DB from the Heroku snapshot:
  `manage.py import_heroku_backup ~/Documents/Fantasy_Website/heroku_backups/dunn-right-fantasy.dump`

## Deploy / config
- `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS` are env-driven
  (`DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`).
- Production settings: `fantasy_app/heroku.py` (set
  `DJANGO_SETTINGS_MODULE=fantasy_app.heroku`). Heroku app: `dunn-right-fantasy`
  (renamed from `fantasy-draft-order` July 2026) at
  https://dunn-right-fantasy-048860865119.herokuapp.com — Basic dyno +
  Essential-0 Postgres. `dunn-right-fantasy-old` is the retired original app
  (empty, $0), kept only as a name placeholder.
- Google sign-in needs `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` env vars.
- No Redis: `collect_yahoo_data` falls back to synchronous collection
  (by design, to avoid a Redis add-on cost).

## Known issue
- `yahoo_collector.py` sets `Projected Result` = actual `Result`, so the Luck
  Report's "vs Projection" column would read 0 for live-collected data. The
  primary "scoring luck" metric (all-play) is unaffected.
