from django.urls import path
from .views import (upload_csv, team_chart, team_detail, team_performance_list, charts_view,
                    team_performance_view, box_plots_filter, stats_charts, stats_charts_filter,
                    stats_charts_filter_less_than, versus)

urlpatterns = [
    path('upload/', upload_csv, name='upload_csv'),
    # path('teams/', team_performance_list, name='team_performance_list'),
    # path('teams/<int:team_id>/', team_detail, name='team_detail'),
    path('charts/teams/', team_chart, name='team_chart'),
    # path('', charts_view, name='charts_view'),
    # path('', team_performance_view, name='team_performance'),
    path('', box_plots_filter, name='team_chart_filter'),
    path('stats/', stats_charts, name='stats_charts'),
    path('stats/filter/', stats_charts_filter, name='stats_charts_filter'),
    path('stats/filter/less_than/', stats_charts_filter_less_than, name='stats_charts_filter_less_than'),
    path('stats/versus/', versus, name='versus')
]
