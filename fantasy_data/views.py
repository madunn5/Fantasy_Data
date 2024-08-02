import json

import pandas as pd
import matplotlib
from django.db.models import Count
from django.template.loader import render_to_string

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .models import TeamPerformance
from django.conf import settings
import plotly.express as px
import plotly.graph_objects as go
from django.shortcuts import redirect
from django.contrib import messages
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def upload_csv(request):
    if request.method == 'POST':
        csv_file = request.FILES['file']
        if not csv_file.name.endswith('.csv'):
            return HttpResponse('This is not a CSV file.')

        # Clear previous data
        TeamPerformance.objects.all().delete()
        logger.info("Previous data successfully deleted.")

        data = pd.read_csv(csv_file)
        data = data.sort_values(by=['Total'], ascending=True)

        data_total = data.groupby(['Team', 'Week']).sum().reset_index()
        data_total = data_total.sort_values(by=['Total'], ascending=True)

        def combine_teams_and_positions(row):
            return f"{row['Team']}"

        data_total['Team'] = data_total.apply(combine_teams_and_positions, axis=1)

        for _, row in data_total.iterrows():
            team_performance, created = TeamPerformance.objects.update_or_create(
                team_name=row['Team'],
                week=row['Week'],
                defaults={
                    'qb_points': row['QB_Points'],
                    'wr_points': row['WR_Points'],
                    'wr_points_total': row['WR_Points_Total'],
                    'rb_points': row['RB_Points'],
                    'rb_points_total': row['RB_Points_Total'],
                    'te_points': row['TE_Points'],
                    'te_points_total': row['TE_Points_Total'],
                    'k_points': row['K_Points'],
                    'def_points': row['DEF_Points'],
                    'total_points': row['Total'],
                    'expected_total': row['Expected Total'],
                    'difference': row['Difference'],
                    'points_against': row['Points Against'],
                    'projected_wins': row.get('Projected Wins', 0),
                    'actual_wins': row.get('Actual Wins', 0),
                    'wins_diff': row.get('Wins Over/(Wins Below)', 0),
                    'result': row.get('Result', 'N/A'),
                    'opponent': row.get('Opponent', 'N/A')
                }
            )

        # Add a success message
        messages.success(request, 'CSV file uploaded and data processed successfully.')

        # Redirect back to the upload page
        return redirect('upload_csv')

    return render(request, 'fantasy_data/upload_csv.html')


def team_performance_view(request):
    data = TeamPerformance.objects.all().values()
    generate_charts(data)
    return render(request, 'fantasy_data/team_performance.html')


def team_performance_list(request):
    teams = TeamPerformance.objects.all()
    return render(request, 'fantasy_data/team_performance_list.html', {'teams': teams})


def team_detail(request, team_id):
    team = TeamPerformance.objects.get(id=team_id)
    return render(request, 'fantasy_data/team_detail.html', {'team': team})


def team_chart(request):
    teams = TeamPerformance.objects.values_list('team_name', flat=True).distinct()
    selected_team = request.GET.get('team', 'all')
    only_box_plots = request.GET.get('only_box_plots', 'false') == 'true'

    # Load data
    team_performance = TeamPerformance.objects.all()
    df = pd.DataFrame(list(team_performance.values()))
    df_sorted = df.sort_values(by=['week'])

    # Filter data based on selected team for box plots
    if selected_team != 'all':
        df_sorted = df_sorted[df_sorted['team_name'] == selected_team]

    # Create the box plots
    fig_points = px.box(df_sorted, x='result', y='total_points', color='result', title='Total Points by Result')
    chart_points = fig_points.to_html(full_html=False)

    fig_wr = px.box(df_sorted, x='result', y='wr_points', color='result', title='Total WR Points by Result')
    chart_wr = fig_wr.to_html(full_html=False)

    fig_qb = px.box(df_sorted, x='result', y='qb_points', color='result', title='Total QB Points by Result')
    chart_qb = fig_qb.to_html(full_html=False)

    fig_rb = px.box(df_sorted, x='result', y='rb_points', color='result', title='Total RB Points by Result')
    chart_rb = fig_rb.to_html(full_html=False)

    fig_te = px.box(df_sorted, x='result', y='te_points', color='result', title='Total TE Points by Result')
    chart_te = fig_te.to_html(full_html=False)

    fig_k = px.box(df_sorted, x='result', y='k_points', color='result', title='Total K Points by Result')
    chart_k = fig_k.to_html(full_html=False)

    fig_def = px.box(df_sorted, x='result', y='def_points', color='result', title='Total DEF Points by Result')
    chart_def = fig_def.to_html(full_html=False)

    context = {
        'teams': teams,
        'selected_team': selected_team,
        'chart_points': chart_points,
        'chart_wr': chart_wr,
        'chart_qb': chart_qb,
        'chart_rb': chart_rb,
        'chart_te': chart_te,
        'chart_k': chart_k,
        'chart_def': chart_def
    }

    if only_box_plots:
        return render(request, 'fantasy_data/partial_box_plots.html', context)

    # Create the regular charts
    fig = px.bar(df_sorted, x='team_name', y='total_points', color='week', title='Total Points by Team Each Week')
    chart = fig.to_html(full_html=False)

    fig_wr_points = px.bar(df_sorted, x='team_name', y='wr_points_total', color='week',
                           title='Total WR Points by Team Each Week')
    chart_wr_points = fig_wr_points.to_html(full_html=False)

    fig_qb_points = px.bar(df_sorted, x='team_name', y='qb_points', color='week',
                           title='Total QB Points by Team Each Week')
    chart_qb_points = fig_qb_points.to_html(full_html=False)

    fig_rb_points = px.bar(df_sorted, x='team_name', y='rb_points_total', color='week',
                           title='Total RB Points by Team Each Week')
    chart_rb_points = fig_rb_points.to_html(full_html=False)

    fig_te_points = px.bar(df_sorted, x='team_name', y='te_points_total', color='week',
                           title='Total TE Points by Team Each Week')
    chart_te_points = fig_te_points.to_html(full_html=False)

    fig_k_points = px.bar(df_sorted, x='team_name', y='k_points', color='week',
                          title='Total K Points by Team Each Week')
    chart_k_points = fig_k_points.to_html(full_html=False)

    fig_def_points = px.bar(df_sorted, x='team_name', y='def_points', color='week',
                            title='Total DEF Points by Team Each Week')
    chart_def_points = fig_def_points.to_html(full_html=False)

    context.update({
        'chart': chart,
        'chart_wr_points': chart_wr_points,
        'chart_qb_points': chart_qb_points,
        'chart_rb_points': chart_rb_points,
        'chart_te_points': chart_te_points,
        'chart_k_points': chart_k_points,
        'chart_def_points': chart_def_points,
    })

    return render(request, 'fantasy_data/team_chart.html', context)


def generate_charts(data):
    # Create the directory for the charts if it doesn't exist
    chart_dir = os.path.join(settings.MEDIA_ROOT, 'charts')
    if not os.path.exists(chart_dir):
        os.makedirs(chart_dir)

    data = list(data.values())
    data = pd.DataFrame(data)

    # Define a color palette for the box plots
    boxplot_palette = {'L': 'lightblue', 'W': 'orange'}

    # Example Chart: Total Points Distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x='team_name', y='total_points', data=data)
    plt.xticks(rotation=90)
    plt.title('Total Points Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'total_points_distribution.png'))
    plt.clf()

    # Additional charts can be added in a similar way:
    # Example: Expected vs Actual Wins
    plt.figure(figsize=(10, 6))
    sns.barplot(x='team_name', y='projected_wins', data=data, color='blue', label='Projected Wins')
    sns.barplot(x='team_name', y='actual_wins', data=data, color='red', label='Actual Wins')
    plt.xticks(rotation=90)
    plt.title('Projected vs Actual Wins')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'projected_vs_actual_wins.png'))
    plt.clf()

    # Example: Points Against Distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x='team_name', y='points_against', data=data)
    plt.xticks(rotation=90)
    plt.title('Points Against Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'points_against_distribution.png'))
    plt.clf()

    # plt.figure(figsize=(10, 6))
    sns.boxplot(x='result', y='total_points', data=data, palette=boxplot_palette)
    # plt.xticks(rotation=90)
    plt.title('Total Points By Result')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'total_points_by_result.png'))
    plt.clf()

    # plt.figure(figsize=(10, 6))
    sns.boxplot(x='result', y='wr_points', data=data, palette=boxplot_palette)
    # plt.xticks(rotation=90)
    plt.title('Total WR Points By Result')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'total_wr_points_by_result.png'))
    plt.clf()

    # plt.figure(figsize=(10, 6))
    sns.boxplot(x='result', y='qb_points', data=data, palette=boxplot_palette)
    # plt.xticks(rotation=90)
    plt.title('Total QB Points By Result')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'total_qb_points_by_result.png'))
    plt.clf()

    # plt.figure(figsize=(10, 6))
    sns.boxplot(x='result', y='rb_points', data=data, palette=boxplot_palette)
    # plt.xticks(rotation=90)
    plt.title('Total RB Points By Result')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'total_rb_points_by_result.png'))
    plt.clf()

    # plt.figure(figsize=(10, 6))
    sns.boxplot(x='result', y='te_points', data=data, palette=boxplot_palette)
    # plt.xticks(rotation=90)
    plt.title('Total TE Points By Result')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'total_te_points_by_result.png'))
    plt.clf()

    # plt.figure(figsize=(10, 6))
    sns.boxplot(x='result', y='k_points', data=data, palette=boxplot_palette)
    # plt.xticks(rotation=90)
    plt.title('Total K Points By Result')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'total_k_points_by_result.png'))
    plt.clf()

    # plt.figure(figsize=(10, 6))
    sns.boxplot(x='result', y='def_points', data=data, palette=boxplot_palette)
    # plt.xticks(rotation=90)
    plt.title('Total DEF Points By Result')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'total_def_points_by_result.png'))
    plt.clf()


def charts_view(request):
    data = TeamPerformance.objects.all().values()
    generate_charts(data)
    return render(request, 'fantasy_data/charts.html')


def box_plots_filter(request):
    teams = TeamPerformance.objects.values_list('team_name', flat=True).distinct()

    # Load data
    team_performance = TeamPerformance.objects.all()
    df = pd.DataFrame(list(team_performance.values()))

    # Create the box plot with dropdown filter
    fig = go.Figure()

    # Add traces for each team
    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['total_points'],
            name=team
        ))

    # Update layout with dropdown menu
    fig.update_layout(
        title='Total Points by Result with Team Filter',
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': 'All Teams',
                        'method': 'update',
                        'args': [{'visible': [True] * len(teams)}, {'title': 'Total Points by Result: All Teams'}]
                    },
                    *[
                        {
                            'label': team,
                            'method': 'update',
                            'args': [{'visible': [team == t for t in teams]},
                                     {'title': f'Total Points by Result: {team}'}]
                        } for team in teams
                    ]
                ],
                'direction': 'down',
                'showactive': True
            }
        ]
    )

    chart = fig.to_html(full_html=False)

    # Create the box plot with dropdown filter
    fig_qb_points = go.Figure()

    # Add traces for each team
    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig_qb_points.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['qb_points'],
            name=team
        ))

    # Update layout with dropdown menu
    fig_qb_points.update_layout(
        title='Total QB Points by Result with Team Filter',
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': 'All Teams',
                        'method': 'update',
                        'args': [{'visible': [True] * len(teams)}, {'title': 'Total QB Points by Result: All Teams'}]
                    },
                    *[
                        {
                            'label': team,
                            'method': 'update',
                            'args': [{'visible': [team == t for t in teams]},
                                     {'title': f'Total QB Points by Result: {team}'}]
                        } for team in teams
                    ]
                ],
                'direction': 'down',
                'showactive': True
            }
        ]
    )

    chart_qb = fig_qb_points.to_html(full_html=False)

    # Create the box plot with dropdown filter
    fig_rb_points = go.Figure()

    # Add traces for each team
    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig_rb_points.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['rb_points'],
            name=team
        ))

    # Update layout with dropdown menu
    fig_rb_points.update_layout(
        title='Total RB Points by Result with Team Filter',
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': 'All Teams',
                        'method': 'update',
                        'args': [{'visible': [True] * len(teams)}, {'title': 'Total RB Points by Result: All Teams'}]
                    },
                    *[
                        {
                            'label': team,
                            'method': 'update',
                            'args': [{'visible': [team == t for t in teams]},
                                     {'title': f'Total RB Points by Result: {team}'}]
                        } for team in teams
                    ]
                ],
                'direction': 'down',
                'showactive': True
            }
        ]
    )

    chart_rb = fig_rb_points.to_html(full_html=False)

    # Create the box plot with dropdown filter
    fig_wr_points = go.Figure()

    # Add traces for each team
    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig_wr_points.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['wr_points'],
            name=team
        ))

    # Update layout with dropdown menu
    fig_wr_points.update_layout(
        title='Total WR Points by Result with Team Filter',
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': 'All Teams',
                        'method': 'update',
                        'args': [{'visible': [True] * len(teams)}, {'title': 'Total WR Points by Result: All Teams'}]
                    },
                    *[
                        {
                            'label': team,
                            'method': 'update',
                            'args': [{'visible': [team == t for t in teams]},
                                     {'title': f'Total WR Points by Result: {team}'}]
                        } for team in teams
                    ]
                ],
                'direction': 'down',
                'showactive': True
            }
        ]
    )

    chart_wr = fig_wr_points.to_html(full_html=False)

    # Create the box plot with dropdown filter
    fig_te_points = go.Figure()

    # Add traces for each team
    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig_te_points.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['te_points'],
            name=team
        ))

    # Update layout with dropdown menu
    fig_te_points.update_layout(
        title='Total TE Points by Result with Team Filter',
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': 'All Teams',
                        'method': 'update',
                        'args': [{'visible': [True] * len(teams)}, {'title': 'Total TE Points by Result: All Teams'}]
                    },
                    *[
                        {
                            'label': team,
                            'method': 'update',
                            'args': [{'visible': [team == t for t in teams]},
                                     {'title': f'Total TE Points by Result: {team}'}]
                        } for team in teams
                    ]
                ],
                'direction': 'down',
                'showactive': True
            }
        ]
    )

    chart_te = fig_te_points.to_html(full_html=False)

    # Create the box plot with dropdown filter
    fig_k_points = go.Figure()

    # Add traces for each team
    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig_k_points.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['k_points'],
            name=team
        ))

    # Update layout with dropdown menu
    fig_k_points.update_layout(
        title='Total K Points by Result with Team Filter',
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': 'All Teams',
                        'method': 'update',
                        'args': [{'visible': [True] * len(teams)}, {'title': 'Total K Points by Result: All Teams'}]
                    },
                    *[
                        {
                            'label': team,
                            'method': 'update',
                            'args': [{'visible': [team == t for t in teams]},
                                     {'title': f'Total K Points by Result: {team}'}]
                        } for team in teams
                    ]
                ],
                'direction': 'down',
                'showactive': True
            }
        ]
    )

    chart_k = fig_k_points.to_html(full_html=False)

    # Create the box plot with dropdown filter
    fig_def_points = go.Figure()

    # Add traces for each team
    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig_def_points.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['def_points'],
            name=team
        ))

    # Update layout with dropdown menu
    fig_def_points.update_layout(
        title='Total DEF Points by Result with Team Filter',
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': 'All Teams',
                        'method': 'update',
                        'args': [{'visible': [True] * len(teams)}, {'title': 'Total DEF Points by Result: All Teams'}]
                    },
                    *[
                        {
                            'label': team,
                            'method': 'update',
                            'args': [{'visible': [team == t for t in teams]},
                                     {'title': f'Total DEF Points by Result: {team}'}]
                        } for team in teams
                    ]
                ],
                'direction': 'down',
                'showactive': True
            }
        ]
    )

    chart_def = fig_def_points.to_html(full_html=False)

    context = {
        'chart': chart,
        'chart_qb': chart_qb,
        'chart_rb': chart_rb,
        'chart_wr': chart_wr,
        'chart_te': chart_te,
        'chart_k': chart_k,
        'chart_def': chart_def
    }

    return render(request, 'fantasy_data/team_chart_filter.html', context)


def stats_charts(request):
    # Retrieve data from the database
    data = TeamPerformance.objects.all().values()

    if not data:
        print("No data retrieved from TeamPerformance model.")
        context = {
            'average_differential_table': "<p>No data available for average differential.</p>",
            'average_by_team_table': "<p>No data available for average by team.</p>"
        }
        return render(request, 'fantasy_data/stats.html', context)

    # Convert data to DataFrame
    data = list(data)
    data = pd.DataFrame(data)

    # Compute average differential
    if 'team_name' in data.columns and 'difference' in data.columns:
        data_avg_differential = data[['team_name', 'difference']]
        data_avg_differential = data_avg_differential.groupby('team_name').mean()
        data_avg_differential = data_avg_differential.sort_values(by=['difference'], ascending=False).reset_index()
        data_avg_differential.index = data_avg_differential.index + 1
        average_differential_table = data_avg_differential.to_html(classes='table table-striped', index=False)
    else:
        average_differential_table = "<p>Columns 'team_name' or 'difference' not found in data.</p>"

    # Compute average by team
    if 'team_name' in data.columns and 'total_points' in data.columns and 'points_against' in data.columns:
        data_avg_by_team = data[['team_name', 'total_points', 'points_against']]
        data_avg_by_team = data_avg_by_team.groupby('team_name').mean()
        data_avg_by_team = data_avg_by_team.sort_values(by=['total_points'], ascending=False).reset_index()
        data_avg_by_team['Diff'] = data_avg_by_team['total_points'] - data_avg_by_team['points_against']
        data_avg_by_team.index = data_avg_by_team.index + 1
        average_by_team_table = data_avg_by_team.to_html(classes='table table-striped', index=False)
    else:
        average_by_team_table = "<p>Columns 'team_name', 'total_points', or 'points_against' not found in data.</p>"

    # Prepare context with the HTML tables
    context = {
        'average_differential_table': average_differential_table,
        'average_by_team_table': average_by_team_table
    }

    html_content = render_to_string('fantasy_data/stats.html', context)
    return HttpResponse(html_content)


def stats_charts_filter(request):
    # Get selected points category
    selected_category = request.GET.get('points_category')

    # Filter data based on user input
    filter_value = request.GET.get('filter_value')
    if filter_value:
        filter_value = float(filter_value)
        filtered_data = TeamPerformance.objects.filter(**{f"{selected_category}__gte": filter_value})
    else:
        filtered_data = TeamPerformance.objects.all()

    # Count occurrences per team for the selected category
    result = filtered_data.values('team_name').annotate(count=Count('id')).order_by('-count')

    # Count occurrences per team where the result is 'W' (win)
    wins = filtered_data.filter(result='W').values('team_name').annotate(win_count=Count('id'))

    # Prepare data for Chart.js
    labels = [item['team_name'] for item in result]
    counts = [item['count'] for item in result]

    # Get win counts for each team and match it with the respective label
    win_counts = [next((w['win_count'] for w in wins if w['team_name'] == label), 0) for label in labels]

    # Render HTML if requested via AJAX, otherwise return JSON
    if 'ajax' in request.GET:
        return JsonResponse({'labels': labels, 'counts': counts, 'winCounts': win_counts})
    else:
        context = {'labels': labels, 'counts': counts, 'winCounts': win_counts}
        return render(request, 'fantasy_data/stats_filter.html', context)


def stats_charts_filter_less_than(request):
    # Get selected points category
    selected_category = request.GET.get('points_category')

    # Filter data based on user input
    filter_value = request.GET.get('filter_value')
    if filter_value:
        filter_value = float(filter_value)
        filtered_data = TeamPerformance.objects.filter(**{f"{selected_category}__lte": filter_value})
    else:
        filtered_data = TeamPerformance.objects.all()

    # Count occurrences per team for the selected category
    result = filtered_data.values('team_name').annotate(count=Count('id')).order_by('-count')

    # Count occurrences per team where the result is 'W' (win)
    wins = filtered_data.filter(result='W').values('team_name').annotate(win_count=Count('id'))

    # Prepare data for Chart.js
    labels = [item['team_name'] for item in result]
    counts = [item['count'] for item in result]

    # Get win counts for each team and match it with the respective label
    win_counts = [next((w['win_count'] for w in wins if w['team_name'] == label), 0) for label in labels]

    # Render HTML if requested via AJAX, otherwise return JSON
    if 'ajax' in request.GET:
        return JsonResponse({'labels': labels, 'counts': counts, 'winCounts': win_counts})
    else:
        context = {'labels': labels, 'counts': counts, 'winCounts': win_counts}
        return render(request, 'fantasy_data/stats_filter_less_than.html', context)
