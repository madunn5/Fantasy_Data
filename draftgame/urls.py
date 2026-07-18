from django.urls import path
from . import views

app_name = 'draftgame'

urlpatterns = [
    path('', views.home, name='home'),
    path('punishment-grid/', views.punishment_grid, name='punishment_grid'),
    path('finalize-grid/', views.toggle_grid_finalized, name='toggle_grid_finalized'),
    path('toggle-voting/', views.toggle_voting, name='toggle_voting'),
    path('punishment-wheel/', views.punishment_wheel, name='punishment_wheel'),
    path('draft-lottery/', views.draft_lottery, name='draft_lottery'),
    path('bucket-of-death/', views.bucket_of_death, name='bucket_of_death'),
    path('bucket-rummage/', views.bucket_rummage, name='bucket_rummage'),
    path('bucket-decision/', views.bucket_decision, name='bucket_decision'),
    path('bucket-must-keep/', views.bucket_rummage_must_keep, name='bucket_rummage_must_keep'),
    path('bucket-special-team/', views.bucket_special_team, name='bucket_special_team'),
    path('bucket-results-admin/', views.bucket_results_admin, name='bucket_results_admin'),
    # Punishment submission + voting (2026 season)
    path('submit-punishment/', views.submit_punishment, name='submit_punishment'),
    path('suggested-so-far/', views.suggestion_list, name='suggestion_list'),
    path('punishment-vote/', views.punishment_vote, name='punishment_vote'),
    path('punishment-results/', views.punishment_results, name='punishment_results'),
    path('punishment-history/', views.punishment_history, name='punishment_history'),
    # register/logout are mounted at the project level under /accounts/ so the
    # 'register' and 'logout' URL names stay global for both halves of the site.
]
