"""Site-wide fantasy-season (year) navigation.

A single helper resolves the "active year" for any request, and a context
processor injects that year plus the list of available years into every
template so the navbar can render one global year picker whose selection
persists across pages.

Precedence for the active year:
  1. ?year=<year> in the querystring (validated, then remembered in the session)
  2. the year stored in the session from a previous page
  3. the most recent season we have data for

Modeled on the season selector in the Fantasy_Draft_Game project.
"""
from .models import TeamPerformance

DEFAULT_YEAR = 2023


def available_years(request=None):
    """Distinct seasons we have team data for, newest first.

    Memoized on the request so the view and the context processor don't each
    hit the database within a single request.
    """
    if request is not None and hasattr(request, "_fd_years"):
        return request._fd_years
    years = list(
        TeamPerformance.objects.values_list("year", flat=True)
        .distinct()
        .order_by("-year")
    )
    if request is not None:
        request._fd_years = years
    return years


def get_selected_year(request, years=None):
    """Resolve the active fantasy year for this request.

    Returns ``(selected_year, years)`` where ``years`` is newest-first. Pass an
    explicit ``years`` list to scope the picker to a different model's seasons
    (e.g. roster years); otherwise it defaults to team-performance seasons.
    """
    if years is None:
        years = available_years(request)
    valid = set(years)
    latest = years[0] if years else DEFAULT_YEAR

    requested = request.GET.get("year")
    if requested is not None:
        try:
            requested_year = int(requested)
        except (TypeError, ValueError):
            requested_year = None
        if requested_year in valid:
            request.session["view_year"] = requested_year

    stored = request.session.get("view_year")
    selected_year = stored if stored in valid else latest
    return selected_year, years


def year_selector(request):
    """Inject the global year-picker state into every template.

    Fails soft (empty dict) so admin/auth/static pages render even if the data
    tables aren't available yet.
    """
    try:
        selected_year, years = get_selected_year(request)
    except Exception:
        return {}
    return {
        "years": years,
        "selected_year": selected_year,
        "view_year": selected_year,
    }
