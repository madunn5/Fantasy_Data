import json
import re

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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

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
                    'projected_wins': row['Projected Result'],
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
    teams = TeamPerformance.objects.values_list('team_name', flat=True).distinct().order_by('team_name')
    selected_team = request.GET.get('team', 'all')
    only_box_plots = request.GET.get('only_box_plots', 'false') == 'true'

    # Load data
    team_performance = TeamPerformance.objects.all()
    df = pd.DataFrame(list(team_performance.values()))

    # Define a function to extract the numeric part
    def extract_week_number(week_str):
        match = re.search(r'\d+', week_str)
        return int(match.group()) if match else 0

    # Apply the function to create a new column
    df['week_number'] = df['week'].apply(extract_week_number)

    # Sort by the new column and drop it if not needed
    df_sorted = df.sort_values(by='week_number').drop(columns=['week_number'])

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
    teams = TeamPerformance.objects.values_list('team_name', flat=True).distinct().order_by('team_name')

    # Load data
    team_performance = TeamPerformance.objects.all()
    df = pd.DataFrame(list(team_performance.values()))

    # Create the box plot for total points
    fig = go.Figure()

    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['total_points'],
            name=team
        ))

    fig.update_layout(
        title='Total Points by Result',
        xaxis_title='Result',
        yaxis_title='Total Points'
    )

    chart = fig.to_html(full_html=False)

    # Create the box plot for QB points
    fig_qb_points = go.Figure()

    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig_qb_points.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['qb_points'],
            name=team
        ))

    fig_qb_points.update_layout(
        title='Total QB Points by Result',
        xaxis_title='Result',
        yaxis_title='QB Points'
    )

    chart_qb = fig_qb_points.to_html(full_html=False)

    # Create the box plot for RB points
    fig_rb_points = go.Figure()

    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig_rb_points.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['rb_points'],
            name=team
        ))

    fig_rb_points.update_layout(
        title='Total RB Points by Result',
        xaxis_title='Result',
        yaxis_title='RB Points'
    )

    chart_rb = fig_rb_points.to_html(full_html=False)

    # Create the box plot for WR points
    fig_wr_points = go.Figure()

    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig_wr_points.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['wr_points'],
            name=team
        ))

    fig_wr_points.update_layout(
        title='Total WR Points by Result',
        xaxis_title='Result',
        yaxis_title='WR Points'
    )

    chart_wr = fig_wr_points.to_html(full_html=False)

    # Create the box plot for TE points
    fig_te_points = go.Figure()

    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig_te_points.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['te_points'],
            name=team
        ))

    fig_te_points.update_layout(
        title='Total TE Points by Result',
        xaxis_title='Result',
        yaxis_title='TE Points'
    )

    chart_te = fig_te_points.to_html(full_html=False)

    # Create the box plot for K points
    fig_k_points = go.Figure()

    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig_k_points.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['k_points'],
            name=team
        ))

    fig_k_points.update_layout(
        title='Total K Points by Result',
        xaxis_title='Result',
        yaxis_title='K Points'
    )

    chart_k = fig_k_points.to_html(full_html=False)

    # Create the box plot for DEF points
    fig_def_points = go.Figure()

    for team in teams:
        filtered_df = df[df['team_name'] == team]
        fig_def_points.add_trace(go.Box(
            x=filtered_df['result'],
            y=filtered_df['def_points'],
            name=team
        ))

    fig_def_points.update_layout(
        title='Total DEF Points by Result',
        xaxis_title='Result',
        yaxis_title='DEF Points'
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
            'average_by_team_table': "<p>No data available for average by team.</p>",
            'wins_table': "<p>No data available for wins above/below projected wins.</p>",
            'max_points_table': "<p>No data available for max points scored in a single game.</p>"
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

    # Compute average points for and against by team
    if 'team_name' in data.columns and 'total_points' in data.columns and 'points_against' in data.columns:
        data_avg_by_team = data[['team_name', 'total_points', 'points_against']]
        data_avg_by_team = data_avg_by_team.groupby('team_name').mean()
        data_avg_by_team = data_avg_by_team.sort_values(by=['total_points'], ascending=False).reset_index()
        data_avg_by_team['Diff'] = data_avg_by_team['total_points'] - data_avg_by_team['points_against']
        data_avg_by_team.index = data_avg_by_team.index + 1
        average_by_team_table = data_avg_by_team.to_html(classes='table table-striped', index=False)
    else:
        average_by_team_table = "<p>Columns 'team_name', 'total_points', or 'points_against' not found in data.</p>"

    # Wins Above/Below Projected Wins
    if 'team_name' in data.columns and 'projected_wins' in data.columns and 'result' in data.columns:
        data_wins = data[['team_name', 'projected_wins', 'result']]
        data_wins = data_wins.groupby('team_name').agg(
            projected_result_count=('projected_wins', lambda x: (x == 'W').sum()),
            result_count=('result', lambda x: (x == 'W').sum())
        ).reset_index()

        # Rename columns for clarity
        data_wins.columns = ['team_name', 'projected_result_count', 'result_count']
        data_wins['Wins Over/(Wins Below)'] = data_wins['result_count'] - data_wins['projected_result_count']
        data_wins = data_wins.sort_values(by=['Wins Over/(Wins Below)'], ascending=False).reset_index()
        del data_wins['index']
        wins_table = data_wins.to_html(classes='table table-striped', index=False)
    else:
        wins_table = "<p>Columns 'team_name', 'projected_result_count', or 'result_count' not found in data.</p>"

    # Max Points Scored in a Single Game
    if 'team_name' in data.columns and 'total_points' in data.columns:
        data_avg_by_team = data[['team_name', 'week', 'total_points', 'result', 'opponent']]
        max_total_rows = data_avg_by_team.groupby('team_name')['total_points'].idxmax()
        result = data_avg_by_team.loc[max_total_rows, ['team_name', 'week', 'total_points', 'result', 'opponent']]
        result = result.sort_values(by=['total_points'], ascending=False).reset_index(drop=True)
        result.index = result.index + 1
        max_points_table = result.to_html(classes='table table-striped', index=False)
    else:
        max_points_table = ("<p>Columns 'team_name', 'week', 'total_points', 'result', or 'opponent' not found in "
                            "data.</p>")

    # Best Possible Score to Date
    if all(col in data.columns for col in
           ['team_name', 'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total', 'k_points',
            'def_points']):
        data_best_possible = data[
            ['team_name', 'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total', 'k_points',
             'def_points']]
        data_best_possible = data_best_possible.groupby('team_name').max()
        data_best_possible['Total'] = (data_best_possible['qb_points'] + data_best_possible['wr_points_total'] +
                                       data_best_possible['rb_points_total'] + data_best_possible['te_points_total'] +
                                       data_best_possible['k_points'] + data_best_possible['def_points'])
        data_best_possible = data_best_possible.sort_values(by=['Total'], ascending=False).reset_index()
        data_best_possible.index = data_best_possible.index + 1
        best_possible_table = data_best_possible.to_html(classes='table table-striped', index=False)
    else:
        best_possible_table = "<p>Necessary columns not found in data.</p>"

    # Worst Possible Score to Date
    if all(col in data.columns for col in
           ['team_name', 'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total', 'k_points',
            'def_points']):
        data_worst_possible = data[
            ['team_name', 'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total', 'k_points',
             'def_points']]
        data_worst_possible = data_worst_possible.groupby('team_name').min()
        data_worst_possible['Total'] = (data_worst_possible['qb_points'] + data_worst_possible['wr_points_total'] +
                                        data_worst_possible['rb_points_total'] + data_worst_possible[
                                            'te_points_total'] +
                                        data_worst_possible['k_points'] + data_worst_possible['def_points'])
        data_worst_possible = data_worst_possible.sort_values(by=['Total'], ascending=False).reset_index()
        data_worst_possible.index = data_worst_possible.index + 1
        worst_possible_table = data_worst_possible.to_html(classes='table table-striped', index=False)
    else:
        worst_possible_table = "<p>Necessary columns not found in data.</p>"

    # Median Possible Score to Date
    if all(col in data.columns for col in
           ['team_name', 'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total', 'k_points',
            'def_points']):
        data_avg_possible = data[
            ['team_name', 'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total', 'k_points',
             'def_points']]
        data_avg_possible = data_avg_possible.groupby('team_name').median()
        data_avg_possible['Total'] = (data_avg_possible['qb_points'] + data_avg_possible['wr_points_total'] +
                                      data_avg_possible['rb_points_total'] + data_avg_possible[
                                          'te_points_total'] + data_avg_possible['k_points'] + data_avg_possible[
                                          'def_points'])
        data_avg_possible = data_avg_possible.sort_values(by=['Total'], ascending=False).reset_index()
        data_avg_possible.index = data_avg_possible.index + 1
        median_possible_table = data_avg_possible.to_html(classes='table table-striped', index=False)
    else:
        median_possible_table = "<p>Necessary columns not found in data.</p>"

    # Average Points per Week
    if all(col in data.columns for col in
           ['team_name', 'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total', 'k_points',
            'def_points']):
        data_avg_possible = data[
            ['team_name', 'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total', 'k_points',
             'def_points']]
        data_avg_possible = data_avg_possible.groupby('team_name').mean()
        data_avg_possible['Total'] = (data_avg_possible['qb_points'] + data_avg_possible['wr_points_total'] +
                                      data_avg_possible['rb_points_total'] + data_avg_possible[
                                          'te_points_total'] + data_avg_possible['k_points'] + data_avg_possible[
                                          'def_points'])
        data_avg_possible = data_avg_possible.sort_values(by=['Total'], ascending=False).reset_index()
        data_avg_possible.index = data_avg_possible.index + 1
        avg_points_week_table = data_avg_possible.to_html(classes='table table-striped', index=False)
    else:
        avg_points_week_table = "<p>Necessary columns not found in data.</p>"

    # Compute average points for and against by team, grouped by result
    if all(col in data.columns for col in ['team_name', 'total_points', 'points_against', 'result']):
        data_avg_by_team = data[['team_name', 'total_points', 'points_against', 'result']]
        data_avg_by_team = data_avg_by_team.groupby(['team_name', 'result']).mean()
        data_avg_by_team = data_avg_by_team.sort_values(by=['total_points'], ascending=False).reset_index()
        data_avg_by_team['Diff'] = data_avg_by_team['total_points'] - data_avg_by_team['points_against']
        data_avg_by_team.index = data_avg_by_team.index + 1
        average_by_team_table_win_and_loss = data_avg_by_team.to_html(classes='table table-striped', index=False)
    else:
        average_by_team_table_win_and_loss = "<p>Necessary columns not found in data.</p>"

    # Median points for and against by team, grouped by result
    if all(col in data.columns for col in ['team_name', 'total_points', 'points_against', 'result']):
        data_avg_by_team = data[['team_name', 'total_points', 'points_against', 'result']]
        data_avg_by_team = data_avg_by_team.groupby(['team_name', 'result']).median()
        data_avg_by_team = data_avg_by_team.sort_values(by=['total_points'], ascending=False).reset_index()
        data_avg_by_team['Diff'] = data_avg_by_team['total_points'] - data_avg_by_team['points_against']
        data_avg_by_team.index = data_avg_by_team.index + 1
        median_by_team_result_table = data_avg_by_team.to_html(classes='table table-striped', index=False)
    else:
        median_by_team_result_table = "<p>Necessary columns not found in data.</p>"

    # Median points for and against by team
    if all(col in data.columns for col in ['team_name', 'total_points', 'points_against']):
        data_avg_by_team = data[['team_name', 'total_points', 'points_against']]
        data_avg_by_team = data_avg_by_team.groupby('team_name').median()
        data_avg_by_team = data_avg_by_team.sort_values(by=['total_points'], ascending=False).reset_index()
        data_avg_by_team['Diff'] = data_avg_by_team['total_points'] - data_avg_by_team['points_against']
        data_avg_by_team.index = data_avg_by_team.index + 1
        median_by_team_table = data_avg_by_team.to_html(classes='table table-striped', index=False)
    else:
        median_by_team_table = "<p>Necessary columns not found in data.</p>"

    # Max points in a single game by team
    if all(col in data.columns for col in ['team_name', 'week', 'total_points', 'result', 'opponent']):
        data_avg_by_team = data[['team_name', 'week', 'total_points', 'result', 'opponent']]
        max_total_rows = data_avg_by_team.groupby('team_name')['total_points'].idxmax()
        result = data_avg_by_team.loc[max_total_rows, ['team_name', 'week', 'total_points', 'result', 'opponent']]
        result = result.sort_values(by=['total_points'], ascending=False).reset_index(drop=True)
        result.index = result.index + 1
        max_points_by_team_table = result.to_html(classes='table table-striped', index=False)
    else:
        max_points_by_team_table = "<p>Necessary columns not found in data.</p>"

    # Max QB points in a single game by team
    if all(col in data.columns for col in ['team_name', 'week', 'qb_points', 'result', 'opponent']):
        data_avg_by_team = data[['team_name', 'week', 'qb_points', 'result', 'opponent']]
        max_total_rows = data_avg_by_team.groupby('team_name')['qb_points'].idxmax()
        result = data_avg_by_team.loc[max_total_rows, ['team_name', 'week', 'qb_points', 'result', 'opponent']]
        result = result.sort_values(by=['qb_points'], ascending=False).reset_index(drop=True)
        result.index = result.index + 1
        max_qb_points_by_team_table = result.to_html(classes='table table-striped', index=False)
    else:
        max_qb_points_by_team_table = "<p>Necessary columns not found in data.</p>"

    # Max WR points in a single game by team
    if all(col in data.columns for col in ['team_name', 'week', 'wr_points_total', 'result', 'opponent']):
        data_avg_by_team = data[['team_name', 'week', 'wr_points_total', 'result', 'opponent']]
        max_total_rows = data_avg_by_team.groupby('team_name')['wr_points_total'].idxmax()
        result = data_avg_by_team.loc[max_total_rows, ['team_name', 'week', 'wr_points_total', 'result', 'opponent']]
        result = result.sort_values(by=['wr_points_total'], ascending=False).reset_index(drop=True)
        result.index = result.index + 1
        max_wr_points_by_team_table = result.to_html(classes='table table-striped', index=False)
    else:
        max_wr_points_by_team_table = "<p>Necessary columns not found in data.</p>"

    # Max RB points in a single game by team
    if all(col in data.columns for col in ['team_name', 'week', 'rb_points_total', 'result', 'opponent']):
        data_avg_by_team = data[['team_name', 'week', 'rb_points_total', 'result', 'opponent']]
        max_total_rows = data_avg_by_team.groupby('team_name')['rb_points_total'].idxmax()
        result = data_avg_by_team.loc[max_total_rows, ['team_name', 'week', 'rb_points_total', 'result', 'opponent']]
        result = result.sort_values(by=['rb_points_total'], ascending=False).reset_index(drop=True)
        result.index = result.index + 1
        max_rb_points_by_team_table = result.to_html(classes='table table-striped', index=False)
    else:
        max_rb_points_by_team_table = "<p>Necessary columns not found in data.</p>"

    # Max TE points in a single game by team
    if all(col in data.columns for col in ['team_name', 'week', 'te_points_total', 'result', 'opponent']):
        data_avg_by_team = data[['team_name', 'week', 'te_points_total', 'result', 'opponent']]
        max_total_rows = data_avg_by_team.groupby('team_name')['te_points_total'].idxmax()
        result = data_avg_by_team.loc[max_total_rows, ['team_name', 'week', 'te_points_total', 'result', 'opponent']]
        result = result.sort_values(by=['te_points_total'], ascending=False).reset_index(drop=True)
        result.index = result.index + 1
        max_te_points_by_team_table = result.to_html(classes='table table-striped', index=False)
    else:
        max_te_points_by_team_table = "<p>Necessary columns not found in data.</p>"

    # Max K points in a single game by team
    if all(col in data.columns for col in ['team_name', 'week', 'k_points', 'result', 'opponent']):
        data_avg_by_team = data[['team_name', 'week', 'k_points', 'result', 'opponent']]
        max_total_rows = data_avg_by_team.groupby('team_name')['k_points'].idxmax()
        result = data_avg_by_team.loc[
            max_total_rows, ['team_name', 'week', 'k_points', 'result', 'opponent']]
        result = result.sort_values(by=['k_points'], ascending=False).reset_index(drop=True)
        result.index = result.index + 1
        max_k_points_by_team_table = result.to_html(classes='table table-striped', index=False)
    else:
        max_k_points_by_team_table = "<p>Necessary columns not found in data.</p>"

    # Max DEF points in a single game by team
    if all(col in data.columns for col in ['team_name', 'week', 'def_points', 'result', 'opponent']):
        data_avg_by_team = data[['team_name', 'week', 'def_points', 'result', 'opponent']]
        max_total_rows = data_avg_by_team.groupby('team_name')['def_points'].idxmax()
        result = data_avg_by_team.loc[
            max_total_rows, ['team_name', 'week', 'def_points', 'result', 'opponent']]
        result = result.sort_values(by=['def_points'], ascending=False).reset_index(drop=True)
        result.index = result.index + 1
        max_def_points_by_team_table = result.to_html(classes='table table-striped', index=False)
    else:
        max_def_points_by_team_table = "<p>Necessary columns not found in data.</p>"

    # Min points in a single game by team
    if all(col in data.columns for col in ['team_name', 'week', 'total_points', 'result', 'opponent']):
        data_avg_by_team = data[['team_name', 'week', 'total_points', 'result', 'opponent']]
        max_total_rows = data_avg_by_team.groupby('team_name')['total_points'].idxmin()
        result = data_avg_by_team.loc[max_total_rows, ['team_name', 'week', 'total_points', 'result', 'opponent']]
        result = result.sort_values(by=['total_points'], ascending=False).reset_index(drop=True)
        result.index = result.index + 1
        min_points_by_team_table = result.to_html(classes='table table-striped', index=False)
    else:
        min_points_by_team_table = "<p>Necessary columns not found in data.</p>"

        # Win/Loss by margin classification
    if all(col in data.columns for col in ['team_name', 'week', 'total_points', 'points_against', 'result']):
        data['point_margin'] = data['total_points'] - data['points_against']

        # Define the categories
        conditions = [
            (data['result'] == 'W') & (data['point_margin'] > 10),
            (data['result'] == 'W') & (data['point_margin'] <= 10),
            (data['result'] == 'L') & (data['point_margin'] < -10),
            (data['result'] == 'L') & (data['point_margin'] >= -10)
        ]
        choices = ['Win > 10', 'Win ≤ 10', 'Loss > 10', 'Loss ≤ 10']

        # Create margin_category column
        data['margin_category'] = np.select(conditions, choices, default='Other')

        # Group by team_name and margin_category to count occurrences
        margin_counts = data.groupby(['team_name', 'margin_category']).size().unstack(fill_value=0)

        # Ensure all columns exist even if some categories are not present for all teams
        for col in ['Win > 10', 'Win ≤ 10', 'Loss > 10', 'Loss ≤ 10']:
            if col not in margin_counts.columns:
                margin_counts[col] = 0

        # Reorder and reset index
        margin_counts = margin_counts[['Win > 10', 'Win ≤ 10', 'Loss > 10', 'Loss ≤ 10']].reset_index()
        margins_counts_table = margin_counts.to_html(classes='table table-striped', index=False)
    else:
        margins_counts_table = "<p>Necessary columns not found in data.</p>"

    # Prepare context with the HTML tables
    context = {
        'average_differential_table': average_differential_table,
        'average_by_team_table': average_by_team_table,
        'wins_table': wins_table,
        'max_points_table': max_points_table,
        'best_possible_table': best_possible_table,
        'worst_possible_table': worst_possible_table,
        'median_possible_table': median_possible_table,
        'avg_points_week_table': avg_points_week_table,
        'average_by_team_table_win_and_loss': average_by_team_table_win_and_loss,
        'median_by_team_result_table': median_by_team_result_table,
        'median_by_team_table': median_by_team_table,
        'max_points_by_team_table': max_points_by_team_table,
        'max_qb_points_by_team_table': max_qb_points_by_team_table,
        'max_wr_points_by_team_table': max_wr_points_by_team_table,
        'max_rb_points_by_team_table': max_rb_points_by_team_table,
        'max_te_points_by_team_table': max_te_points_by_team_table,
        'max_k_points_by_team_table': max_k_points_by_team_table,
        'max_def_points_by_team_table': max_def_points_by_team_table,
        'min_points_by_team_table': min_points_by_team_table,
        'margins_counts_table': margins_counts_table,
    }

    # Render the HTML content using the context
    html_content = render_to_string('fantasy_data/stats.html', context)
    return HttpResponse(html_content)


def stats_charts_filter(request):
    # Get selected points category and comparison operator
    selected_category = request.GET.get('points_category')
    comparison_operator = request.GET.get('comparison_operator', 'gte')  # Default to 'gte' if not specified

    # Filter data based on user input
    filter_value = request.GET.get('filter_value')
    if filter_value:
        filter_value = float(filter_value)
        # Use the comparison operator chosen by the user
        filter_key = f"{selected_category}__{comparison_operator}"
        filtered_data = TeamPerformance.objects.filter(**{filter_key: filter_value})
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


def prepare_data():
    team_performance = TeamPerformance.objects.all()
    df = pd.DataFrame(list(team_performance.values()))

    # Drop rows with missing values in key columns
    df = df.dropna(subset=[
        'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total',
        'k_points', 'def_points', 'total_points', 'result'
    ])

    # Ensure numerical columns are in numeric format
    numeric_columns = [
        'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total',
        'k_points', 'def_points', 'total_points'
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df = df.dropna(subset=numeric_columns)  # Drop rows with NaNs in numeric columns

    # Encode categorical columns
    label_encoder_team = LabelEncoder()
    df['team_name'] = label_encoder_team.fit_transform(df['team_name'].astype(str))
    label_encoder_opponent = LabelEncoder()
    df['opponent'] = label_encoder_opponent.fit_transform(df['opponent'].astype(str))
    label_encoder_result = LabelEncoder()
    df['result'] = label_encoder_result.fit_transform(df['result'].astype(str))

    # Binary outcome: 1 if the team won, 0 otherwise
    df['win'] = df['result'].apply(lambda x: 1 if x == 1 else 0)

    return df, label_encoder_team, label_encoder_opponent, label_encoder_result


def train_model():
    df, label_encoder_team, label_encoder_opponent, label_encoder_result = prepare_data()

    X = df[['qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total', 'k_points', 'def_points',
            'total_points']]
    y = df['win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    return model, label_encoder_team, label_encoder_opponent, label_encoder_result


def versus(request):
    teams = TeamPerformance.objects.values_list('team_name', flat=True).distinct().order_by('team_name')
    team1 = request.GET.get('team1', None)
    team2 = request.GET.get('team2', None)

    if not team1 or not team2:
        return render(request, 'fantasy_data/versus.html', {'teams': teams})

    # Load and prepare data for prediction
    model, label_encoder_team, label_encoder_opponent, label_encoder_result = train_model()

    team_performance = TeamPerformance.objects.filter(team_name__in=[team1, team2])
    df = pd.DataFrame(list(team_performance.values()))

    # Drop rows with missing values and ensure numeric columns are cleaned
    df = df.dropna(subset=[
        'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total',
        'k_points', 'def_points', 'total_points'
    ])
    numeric_columns = [
        'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total',
        'k_points', 'def_points', 'total_points'
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df = df.dropna(subset=numeric_columns)  # Drop rows with NaNs in numeric columns

    if team1 not in label_encoder_team.classes_ or team2 not in label_encoder_team.classes_:
        return render(request, 'fantasy_data/versus.html',
                      {'teams': teams,
                       'error': f"One or both teams '{team1}' and '{team2}' not found in training data."})

    # Prepare the feature vectors for the teams
    df_team1 = df[df['team_name'] == team1].mean(numeric_only=True)
    df_team2 = df[df['team_name'] == team2].mean(numeric_only=True)

    # Calculate the absolute difference between the two feature vectors
    feature_differences = df_team1[numeric_columns] - df_team2[numeric_columns]

    # Prepare the data for the template
    difference_between_teams = []
    for col in numeric_columns:
        difference_between_teams.append({
            'Position': col,
            'team1': df_team1[col],
            'team2': df_team2[col],
            'Difference': feature_differences[col]
        })

    # Prepare the feature vectors for prediction
    X_team1 = pd.DataFrame([df_team1[numeric_columns].values], columns=numeric_columns)
    X_team2 = pd.DataFrame([df_team2[numeric_columns].values], columns=numeric_columns)

    # Predict the probability of winning for each team
    prob_team1 = model.predict_proba(X_team1)[:, 1][0]
    prob_team2 = model.predict_proba(X_team2)[:, 1][0]

    # Normalize probabilities
    total_prob = prob_team1 + prob_team2
    prob_team1_normalized = prob_team1 / total_prob
    prob_team2_normalized = prob_team2 / total_prob

    # Generate prediction messages
    if prob_team1_normalized > prob_team2_normalized:
        prediction = (
            f"{team1} is more likely to win with a probability of {prob_team1_normalized:.2%}. "
            f"{team2} has a probability of {prob_team2_normalized:.2%} to win."
        )
    elif prob_team1_normalized < prob_team2_normalized:
        prediction = (
            f"{team2} is more likely to win with a probability of {prob_team2_normalized:.2%}. "
            f"{team1} has a probability of {prob_team1_normalized:.2%} to win."
        )
    else:
        prediction = (
            "It's too close to call! Both teams have similar chances with "
            f"{team1} at {prob_team1_normalized:.2%} and {team2} at {prob_team2_normalized:.2%}."
        )

    # Create comparison charts
    fig_total_points = px.box(
        df, x='team_name', y='total_points', color='team_name',
        title=f'Total Points Comparison: {team1} vs {team2}'
    )
    chart_total_points = fig_total_points.to_html(full_html=False)

    fig_wr_points = px.box(
        df, x='team_name', y='wr_points_total', color='team_name',
        title=f'WR Points Comparison: {team1} vs {team2}'
    )
    chart_wr_points = fig_wr_points.to_html(full_html=False)

    fig_qb_points = px.box(
        df, x='team_name', y='qb_points', color='team_name',
        title=f'QB Points Comparison: {team1} vs {team2}'
    )
    chart_qb_points = fig_qb_points.to_html(full_html=False)

    fig_rb_points = px.box(
        df, x='team_name', y='rb_points_total', color='team_name',
        title=f'RB Points Comparison: {team1} vs {team2}'
    )
    chart_rb_points = fig_rb_points.to_html(full_html=False)

    fig_te_points = px.box(
        df, x='team_name', y='te_points_total', color='team_name',
        title=f'TE Points Comparison: {team1} vs {team2}'
    )
    chart_te_points = fig_te_points.to_html(full_html=False)

    fig_k_points = px.box(
        df, x='team_name', y='k_points', color='team_name',
        title=f'K Points Comparison: {team1} vs {team2}'
    )
    chart_k_points = fig_k_points.to_html(full_html=False)

    fig_def_points = px.box(
        df, x='team_name', y='def_points', color='team_name',
        title=f'DEF Points Comparison: {team1} vs {team2}'
    )
    chart_def_points = fig_def_points.to_html(full_html=False)

    context = {
        'teams': teams,
        'team1': team1,
        'team2': team2,
        'chart_total_points': chart_total_points,
        'chart_wr_points': chart_wr_points,
        'chart_qb_points': chart_qb_points,
        'chart_rb_points': chart_rb_points,
        'chart_te_points': chart_te_points,
        'chart_k_points': chart_k_points,
        'chart_def_points': chart_def_points,
        'prediction': prediction,
        'difference_between_teams': difference_between_teams
    }

    return render(request, 'fantasy_data/versus.html', context)


def win_probability_against_all_teams(request):
    teams = TeamPerformance.objects.values_list('team_name', flat=True).distinct().order_by('team_name')
    selected_team = request.GET.get('team', None)

    if not selected_team:
        return render(request, 'fantasy_data/probabilities.html', {'teams': teams})

    # Load and prepare data for prediction
    model, label_encoder_team, label_encoder_opponent, label_encoder_result = train_model()

    team_performance = TeamPerformance.objects.all()
    df = pd.DataFrame(list(team_performance.values()))

    # Drop rows with missing values and ensure numeric columns are cleaned
    df = df.dropna(subset=[
        'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total',
        'k_points', 'def_points', 'total_points'
    ])
    numeric_columns = [
        'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total',
        'k_points', 'def_points', 'total_points'
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df = df.dropna(subset=numeric_columns)  # Drop rows with NaNs in numeric columns

    if selected_team not in label_encoder_team.classes_:
        return render(request, 'fantasy_data/probabilities.html',
                      {'teams': teams,
                       'error': f"Team '{selected_team}' not found in training data."})

    # Prepare the feature vector for the selected team
    df_selected_team = df[df['team_name'] == selected_team].mean(numeric_only=True)

    win_probabilities = []
    for opponent in teams:
        if opponent == selected_team:
            continue

        # Prepare the feature vector for the opponent
        df_opponent = df[df['team_name'] == opponent].mean(numeric_only=True)

        # Calculate the absolute difference between the two feature vectors
        feature_differences = df_selected_team[numeric_columns] - df_opponent[numeric_columns]

        # Prepare the feature vectors for prediction
        X_selected_team = pd.DataFrame([df_selected_team[numeric_columns].values], columns=numeric_columns)
        X_opponent = pd.DataFrame([df_opponent[numeric_columns].values], columns=numeric_columns)

        # Predict the probability of winning for each team
        prob_selected_team = model.predict_proba(X_selected_team)[:, 1][0]
        prob_opponent = model.predict_proba(X_opponent)[:, 1][0]

        # Normalize probabilities
        total_prob = prob_selected_team + prob_opponent
        prob_selected_team_normalized = prob_selected_team / total_prob
        prob_opponent_normalized = prob_opponent / total_prob

        win_probabilities.append({
            'opponent': opponent,
            'selected_team_probability': f"{prob_selected_team_normalized:.2%}",
            'opponent_probability': f"{prob_opponent_normalized:.2%}",
            'difference': feature_differences.to_dict()
        })

    return render(request, 'fantasy_data/probabilities.html', {
        'teams': teams,
        'selected_team': selected_team,
        'win_probabilities': win_probabilities,
    })
