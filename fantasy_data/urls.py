from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_csv, name='upload_csv'),
    path('team-chart/', views.team_chart, name='team_chart'),
    path('team-chart-filter/', views.box_plots_filter, name='team_chart_filter'),
    path('stats/', views.stats_charts, name='stats_charts'),
    path('stats-filter/', views.stats_charts_filter, name='stats_charts_filter'),
    path('top-tens/', views.top_tens, name='top_tens'),
    path('', views.home, name='home'),
    path('versus/', views.versus, name='versus'),
    path('win-probability/', views.win_probability_against_all_teams, name='win_probability_against_all_teams'),
    
    # New visualization URLs
    path('position-contribution/', views.position_contribution_chart, name='position_contribution'),
    path('performance-trend/', views.performance_trend, name='performance_trend'),
    path('win-probability-heatmap/', views.win_probability_heatmap, name='win_probability_heatmap'),
    path('power-rankings/', views.power_rankings, name='power_rankings'),
    path('playoff-odds/', views.playoff_odds, name='playoff_odds'),
    path('luck-report/', views.luck_report, name='luck_report'),
    path('bench-report/', views.bench_report, name='bench_report'),
    
    # Player data URLs
    path('collect-data/', views.collect_yahoo_data, name='collect_yahoo_data'),
    path('debug-team-data/', views.debug_team_data, name='debug_team_data'),
    path('oauth-authorize/', views.oauth_authorize, name='oauth_authorize'),
    path('players/', views.player_list, name='player_list'),
    path('player/<int:player_id>/', views.player_detail, name='player_detail'),
    path('auth/callback/', views.oauth_callback, name='oauth_callback'),
    path('team-owner-mapping/', views.team_owner_mapping, name='team_owner_mapping'),
    path('export-lineups/', views.export_lineups_csv, name='export_lineups_csv'),
]