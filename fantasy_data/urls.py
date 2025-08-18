from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_csv, name='upload_csv'),
    path('team-performance/', views.team_performance_view, name='team_performance'),
    path('team-performance-list/', views.team_performance_list, name='team_performance_list'),
    path('team-detail/<int:team_id>/', views.team_detail, name='team_detail'),
    path('team-chart/', views.team_chart, name='team_chart'),
    path('team-chart-filter/', views.box_plots_filter, name='team_chart_filter'),
    path('charts/', views.charts_view, name='charts'),
    path('stats/', views.stats_charts, name='stats_charts'),
    path('stats-filter/', views.stats_charts_filter, name='stats_charts_filter'),
    path('stats-filter-less-than/', views.stats_charts_filter_less_than, name='stats_charts_filter_less_than'),
    path('top-tens/', views.top_tens, name='top_tens'),
    path('', views.home, name='home'),
    path('versus/', views.versus, name='versus'),
    path('win-probability/', views.win_probability_against_all_teams, name='win_probability_against_all_teams'),
    
    # New visualization URLs
    path('position-contribution/', views.position_contribution_chart, name='position_contribution'),
    path('performance-trend/', views.performance_trend, name='performance_trend'),
    path('win-probability-heatmap/', views.win_probability_heatmap, name='win_probability_heatmap'),
    
    # Player data URLs
    path('collect-data/', views.collect_yahoo_data, name='collect_yahoo_data'),
    path('debug-team-data/', views.debug_team_data, name='debug_team_data'),
    path('oauth-authorize/', views.oauth_authorize, name='oauth_authorize'),
    path('players/', views.player_list, name='player_list'),
    path('player/<int:player_id>/', views.player_detail, name='player_detail'),
    path('auth/callback/', views.oauth_callback, name='oauth_callback'),
]