import os

from .models import Season


def get_selected_season(request):
    """
    Resolve the site-wide "viewing year".

    Order of precedence:
      1. ?year=<year> in the request (also remembered in the session)
      2. the year previously stored in the session
      3. the most recent season (default on first visit)

    Returns (season, all_seasons_ascending). season is None only when no
    seasons exist at all.
    """
    all_seasons = list(Season.objects.all().order_by('year'))
    if not all_seasons:
        return None, []

    by_year = {s.year: s for s in all_seasons}

    requested = request.GET.get('year')
    if requested:
        try:
            year = int(requested)
        except (TypeError, ValueError):
            year = None
        if year in by_year:
            request.session['view_year'] = year

    stored = request.session.get('view_year')
    season = by_year.get(stored) or all_seasons[-1]
    return season, all_seasons


# Pages whose content is scoped to the selected season (so the year picker is shown).
YEAR_AWARE_VIEWS = {
    'punishment_grid', 'punishment_history', 'punishment_results',
    'punishment_wheel', 'bucket_of_death', 'bucket_results_admin',
}


def season_has_ended(season):
    """A season has 'ended' once its winning punishment is recorded."""
    return bool(season and season.punishments.filter(is_winner=True).exists())


def selected_season(request):
    """Inject the global year selector state into every template."""
    season, all_seasons = get_selected_season(request)
    url_name = getattr(request.resolver_match, 'url_name', None)
    return {
        'all_seasons': all_seasons,
        'selected_season': season,
        'active_season': Season.get_active(),
        'view_year': season.year if season else None,
        'show_year_bar': url_name in YEAR_AWARE_VIEWS,
        'selected_season_ended': season_has_ended(season),
        'google_auth_enabled': bool(os.environ.get('GOOGLE_CLIENT_ID')),
    }
