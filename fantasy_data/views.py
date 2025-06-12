import json
import re
import os
import logging
import functools
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.db.models import Count, F, Q
from django.template.loader import render_to_string
from django.conf import settings
from django.contrib import messages
from django.core.cache import cache

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .models import TeamPerformance

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define a simple cache decorator for expensive functions
def memoize(timeout=3600):
    """Cache the result of a function for a specified time period"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name and arguments
            key = f"fantasy_cache_{func.__name__}_{str(args)}_{str(kwargs)}"
            result = cache.get(key)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result, timeout)
            return result
        return wrapper
    return decorator


def upload_csv(request):
    if request.method == 'POST':
        csv_file = request.FILES['file']
        if not csv_file.name.endswith('.csv'):
            return HttpResponse('This is not a CSV file.')

        # Get the year from the form
        year = request.POST.get('year', 2023)
        try:
            year = int(year)
        except (ValueError, TypeError):
            year = 2023

        # Clear previous data
        # TeamPerformance.objects.all().delete()
        # logger.info("Previous data successfully deleted.")

        data = pd.read_csv(csv_file)
        data = data.sort_values(by=['Total'], ascending=True)

        data_total = data.groupby(['Team', 'Week']).sum().reset_index()
        data_total = data_total.sort_values(by=['Total'], ascending=True)

        # Bulk create/update for better performance
        bulk_updates = []
        for _, row in data_total.iterrows():
            team_performance, created = TeamPerformance.objects.update_or_create(
                team_name=row['Team'],
                week=row['Week'],
                year=year,
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
                    'opponent': row.get('Opponent', 'N/A'),
                    'margin': row['Margin of Matchup']
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
    # Get available years and selected year
    years = TeamPerformance.objects.values_list('year', flat=True).distinct().order_by('year')
    selected_year = request.GET.get('year', years.last() if years else 2023)
    
    # Convert selected_year to integer
    try:
        selected_year = int(selected_year)
    except (ValueError, TypeError):
        selected_year = years.last() if years else 2023

    # Load data filtered by year
    team_performance = TeamPerformance.objects.filter(year=selected_year)
    
    # Get teams for the selected year
    teams = team_performance.values_list('team_name', flat=True).distinct().order_by('team_name')
    selected_team = request.GET.get('team', 'all')
    only_box_plots = request.GET.get('only_box_plots', 'false') == 'true'
    
    # Only fetch the fields we need
    needed_fields = ['team_name', 'week', 'qb_points', 'wr_points', 'wr_points_total', 
                    'rb_points', 'rb_points_total', 'te_points', 'te_points_total', 
                    'k_points', 'def_points', 'total_points', 'result']
    df = pd.DataFrame(list(team_performance.values(*needed_fields)))
    
    if df.empty:
        context = {
            'teams': teams,
            'selected_team': selected_team,
            'years': years,
            'selected_year': selected_year,
            'error_message': 'No data available for the selected year.'
        }
        return render(request, 'fantasy_data/partial_box_plots.html' if only_box_plots else 'fantasy_data/team_chart.html', context)

    # Compile regex pattern once
    week_pattern = re.compile(r'\d+')
    
    # Extract week number using vectorized operations
    def extract_week_number(week_str):
        match = week_pattern.search(week_str)
        return int(match.group()) if match else 0
    
    # Apply the function to create a new column
    df['week_number'] = df['week'].apply(extract_week_number)

    # Sort by the new column and drop it if not needed
    df.sort_values(by='week_number', inplace=True)
    df_sorted = df.drop(columns=['week_number'])
    
    # Only create a copy if filtering is needed
    if selected_team != 'all':
        df_box_plots = df_sorted[df_sorted['team_name'] == selected_team]
    else:
        df_box_plots = df_sorted

    # Create the box plots using the filtered data
    fig_points = px.box(df_box_plots, x='result', y='total_points', color='result', title='Total Points by Result')
    chart_points = fig_points.to_html(full_html=False)

    fig_wr = px.box(df_box_plots, x='result', y='wr_points', color='result', title='Total WR Points by Result')
    chart_wr = fig_wr.to_html(full_html=False)

    fig_qb = px.box(df_box_plots, x='result', y='qb_points', color='result', title='Total QB Points by Result')
    chart_qb = fig_qb.to_html(full_html=False)

    fig_rb = px.box(df_box_plots, x='result', y='rb_points', color='result', title='Total RB Points by Result')
    chart_rb = fig_rb.to_html(full_html=False)

    fig_te = px.box(df_box_plots, x='result', y='te_points', color='result', title='Total TE Points by Result')
    chart_te = fig_te.to_html(full_html=False)

    fig_k = px.box(df_box_plots, x='result', y='k_points', color='result', title='Total K Points by Result')
    chart_k = fig_k.to_html(full_html=False)

    fig_def = px.box(df_box_plots, x='result', y='def_points', color='result', title='Total DEF Points by Result')
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
        'chart_def': chart_def,
        'years': years,
        'selected_year': selected_year
    }

    if only_box_plots:
        return render(request, 'fantasy_data/partial_box_plots.html', context)

    # Create the regular charts only if needed (not for box_plots only view)
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
        'page_title': f'Team Boxplot Charts ({selected_year})'
    })

    return render(request, 'fantasy_data/team_chart.html', context)


def generate_charts(data):
    # Create the directory for the charts if it doesn't exist
    chart_dir = os.path.join(settings.MEDIA_ROOT, 'charts')
    if not os.path.exists(chart_dir):
        os.makedirs(chart_dir)

    # Only convert to DataFrame if it's not already one
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(list(data))
    
    if data.empty:
        logger.warning("No data available for chart generation")
        return

    # Define a color palette for the box plots
    boxplot_palette = {'L': 'lightblue', 'W': 'orange'}
    
    # Define common chart parameters
    chart_params = {
        'figsize': (10, 6),
        'rotation': 90,
        'chart_dir': chart_dir
    }
    
    # Function to create and save a bar chart
    def create_bar_chart(x, y, title, filename, color=None, label=None):
        plt.figure(figsize=chart_params['figsize'])
        if color and label:
            sns.barplot(x=x, y=y, data=data, color=color, label=label)
        else:
            sns.barplot(x=x, y=y, data=data)
        plt.xticks(rotation=chart_params['rotation'])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(chart_params['chart_dir'], filename))
        plt.clf()
    
    # Function to create and save a box plot
    def create_box_plot(x, y, title, filename):
        sns.boxplot(x=x, y=y, data=data, palette=boxplot_palette)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(chart_params['chart_dir'], filename))
        plt.clf()
    
    # Create bar charts
    create_bar_chart('team_name', 'total_points', 'Total Points Distribution', 'total_points_distribution.png')
    
    # Expected vs Actual Wins
    plt.figure(figsize=chart_params['figsize'])
    sns.barplot(x='team_name', y='projected_wins', data=data, color='blue', label='Projected Wins')
    sns.barplot(x='team_name', y='actual_wins', data=data, color='red', label='Actual Wins')
    plt.xticks(rotation=chart_params['rotation'])
    plt.title('Projected vs Actual Wins')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(chart_params['chart_dir'], 'projected_vs_actual_wins.png'))
    plt.clf()
    
    create_bar_chart('team_name', 'points_against', 'Points Against Distribution', 'points_against_distribution.png')
    
    # Create box plots for different metrics
    metrics = [
        ('total_points', 'Total Points By Result', 'total_points_by_result.png'),
        ('wr_points', 'Total WR Points By Result', 'total_wr_points_by_result.png'),
        ('qb_points', 'Total QB Points By Result', 'total_qb_points_by_result.png'),
        ('rb_points', 'Total RB Points By Result', 'total_rb_points_by_result.png'),
        ('te_points', 'Total TE Points By Result', 'total_te_points_by_result.png'),
        ('k_points', 'Total K Points By Result', 'total_k_points_by_result.png'),
        ('def_points', 'Total DEF Points By Result', 'total_def_points_by_result.png')
    ]
    
    for metric, title, filename in metrics:
        create_box_plot('result', metric, title, filename)


def charts_view(request):
    data = TeamPerformance.objects.all().values()
    generate_charts(data)
    return render(request, 'fantasy_data/charts.html')


def box_plots_filter(request):
    teams = TeamPerformance.objects.values_list('team_name', flat=True).distinct().order_by('team_name')
    
    # Get available years and selected year
    years = TeamPerformance.objects.values_list('year', flat=True).distinct().order_by('year')
    selected_year = request.GET.get('year', years.last() if years else 2023)
    
    # Convert selected_year to integer
    try:
        selected_year = int(selected_year)
    except (ValueError, TypeError):
        selected_year = years.last() if years else 2023

    # Load data filtered by year
    team_performance = TeamPerformance.objects.filter(year=selected_year)
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
        'chart_def': chart_def,
        'years': years,
        'selected_year': selected_year,
        'page_title': f'Boxplot Comparisons ({selected_year})'
    }

    return render(request, 'fantasy_data/team_chart_filter.html', context)


def stats_charts(request):
    # Get available years and selected year
    years = TeamPerformance.objects.values_list('year', flat=True).distinct().order_by('year')
    selected_year = request.GET.get('year', years.last() if years else 2023)
    
    # Convert selected_year to integer
    try:
        selected_year = int(selected_year)
    except (ValueError, TypeError):
        selected_year = years.last() if years else 2023
        
    # Retrieve data from the database filtered by year
    data = TeamPerformance.objects.filter(year=selected_year).values()

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

    # Compute average by team after dropping the two lowest scores for each team
    if 'team_name' in data.columns and 'total_points' in data.columns and 'points_against' in data.columns:
        # Define a function to process each team's data
        def calculate_trimmed_averages(group):
            if len(group) > 2:  # Drop the two lowest scores only if there are more than 2
                group = group.nlargest(len(group) - 2, 'total_points')
            return pd.Series({
                'avg_total_points': group['total_points'].mean(),
                'avg_points_against': group['points_against'].mean()
            })

        # Apply the function to each team
        data_avg_by_team = (
            data.groupby('team_name')
            .apply(calculate_trimmed_averages)
            .reset_index()
        )

        # Calculate the difference between averages
        # data_avg_by_team['Diff'] = data_avg_by_team['avg_total_points'] - data_avg_by_team['avg_points_against']

        # Sort by total points average
        data_avg_by_team = data_avg_by_team.sort_values(by='avg_total_points', ascending=False).reset_index(drop=True)
        data_avg_by_team.index = data_avg_by_team.index + 1

        # Convert to HTML
        average_by_team_table_drop_two_lowest = data_avg_by_team.to_html(classes='table table-striped', index=False)
    else:
        average_by_team_table_drop_two_lowest = "<p>Columns 'team_name', 'total_points', or 'points_against' not found in data.</p>"

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

    # Max Difference points in a single game by team
    if all(col in data.columns for col in ['team_name', 'week', 'difference', 'result', 'opponent']):
        data_avg_by_team = data[['team_name', 'week', 'difference', 'result', 'opponent']]
        max_total_rows = data_avg_by_team.groupby('team_name')['difference'].idxmax()
        result = data_avg_by_team.loc[
            max_total_rows, ['team_name', 'week', 'difference', 'result', 'opponent']]
        result = result.sort_values(by=['difference'], ascending=False).reset_index(drop=True)
        result.index = result.index + 1
        max_difference_points_by_team_table = result.to_html(classes='table table-striped', index=False)
    else:
        max_difference_points_by_team_table = "<p>Necessary columns not found in data.</p>"

    # Max negative difference points in a single game by team
    if all(col in data.columns for col in ['team_name', 'week', 'difference', 'result', 'opponent']):
        data_avg_by_team = data[['team_name', 'week', 'difference', 'result', 'opponent']]
        # Use idxmin to get the most negative 'difference' for each team
        max_total_rows = data_avg_by_team.groupby('team_name')['difference'].idxmin()
        result = data_avg_by_team.loc[max_total_rows, ['team_name', 'week', 'difference', 'result', 'opponent']]
        # Sort by 'difference' in ascending order to show the most negative values first
        result = result.sort_values(by=['difference'], ascending=True).reset_index(drop=True)
        result.index = result.index + 1
        min_difference_points_by_team_table = result.to_html(classes='table table-striped', index=False)
    else:
        min_difference_points_by_team_table = "<p>Necessary columns not found in data.</p>"

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

    # Record if you played the same team each week
    # Retrieve all data from the TeamPerformance model
    # data = TeamPerformance.objects.all().values()
    #
    # # Convert the data to a DataFrame
    # data = pd.DataFrame(data)

    # Check if necessary columns exist
    if all(col in data.columns for col in ['team_name', 'week', 'total_points', 'points_against']):
        # Initialize a list to store team records with percentages
        teams_record_list = []

        # Get the list of unique teams
        teams = data['team_name'].unique()

        # Loop through each team
        for team in teams:
            selected_team_data = data[data['team_name'] == team]
            other_teams_data = data[data['team_name'] != team]

            # Initialize a dictionary to store win-loss records for this team
            team_vs_others_record = {}

            # Initialize total wins and losses counters
            total_wins = 0
            total_losses = 0
            total_ties = 0

            # Loop through each unique opponent for the selected team
            for opponent in other_teams_data['team_name'].unique():
                opponent_data = other_teams_data[other_teams_data['team_name'] == opponent]

                # Initialize win/loss counters for this opponent
                wins = 0
                losses = 0
                ties = 0

                # Loop through each week the selected team and opponent played
                for week in selected_team_data['week'].unique():
                    selected_team_week = selected_team_data[selected_team_data['week'] == week]
                    opponent_week = opponent_data[opponent_data['week'] == week]

                    if not opponent_week.empty:
                        # Compare points and update win/loss record
                        if selected_team_week['total_points'].values[0] > opponent_week['total_points'].values[0]:
                            wins += 1
                        elif selected_team_week['total_points'].values[0] < opponent_week['total_points'].values[0]:
                            losses += 1
                        else:
                            ties += 1

                # Update the total wins and losses
                total_wins += wins
                total_losses += losses
                total_ties += ties

                # Add the result for the opponent
                team_vs_others_record[opponent] = f"{wins}-{losses}-{ties}"

            # Calculate total games and win percentage
            total_games = total_wins + total_losses + total_ties
            win_percentage = (total_wins / total_games * 100) if total_games > 0 else 0

            # Add the team's total win-loss record and percentage to the team record
            team_vs_others_record['Total'] = f"{total_wins}-{total_losses}-{total_ties} ({win_percentage:.2f}%)"

            # Store the team, total record, and win percentage in a tuple
            teams_record_list.append((team, team_vs_others_record, win_percentage))

        # Sort teams by win percentage in descending order
        sorted_teams = sorted(teams_record_list, key=lambda x: x[2], reverse=True)

        # Initialize a string to store the combined HTML for all teams' records
        combined_tables = ""

        # Loop through the sorted teams and generate HTML with ranking
        for rank, (team, team_vs_others_record, win_percentage) in enumerate(sorted_teams, start=1):
            # Convert the team's win-loss records to a DataFrame
            team_record_df = pd.DataFrame(list(team_vs_others_record.items()), columns=['Opponent', 'Record'])

            # Add the ranking to the team's name in the heading
            team_table_html = f"<h3>{team} Record Against Other Teams (Rank: {rank})</h3>" + team_record_df.to_html(
                classes='table table-striped', index=False)

            # Append the team table to the combined_tables string
            combined_tables += team_table_html

    else:
        combined_tables = "<p>Necessary columns not found in data.</p>"

    # Check if necessary columns exist
    if all(col in data.columns for col in ['team_name', 'week', 'total_points', 'points_against', 'result']):
        # Initialize a list to store team records by week
        weekly_record_list = []

        # Get the list of unique weeks
        weeks = sorted(data['week'].unique())

        # Loop through each week
        for week in weeks:
            # Filter data for the current week
            weekly_data = data[data['week'] == week]

            # Initialize a dictionary to store records for all teams in this week
            weekly_team_record = {}

            # Loop through each team
            for team in weekly_data['team_name'].unique():
                # Get the team's data for this week
                team_week_data = weekly_data[weekly_data['team_name'] == team]

                # Fetch the result for the team
                week_result = team_week_data['result'].values[0] if not team_week_data.empty else "N/A"

                # Compare the team's points against all opponents in this week
                wins = 0
                losses = 0
                ties = 0

                for opponent_team in weekly_data['team_name'].unique():
                    if opponent_team != team:
                        opponent_week_data = weekly_data[weekly_data['team_name'] == opponent_team]

                        if not opponent_week_data.empty:
                            # Compare points and update win/loss record
                            if team_week_data['total_points'].values[0] > opponent_week_data['total_points'].values[0]:
                                wins += 1
                            elif team_week_data['total_points'].values[0] < opponent_week_data['total_points'].values[
                                0]:
                                losses += 1
                            else:
                                ties += 1

                # Add the result and record for the team
                total_games = wins + losses + ties
                win_percentage = (wins / total_games * 100) if total_games > 0 else 0
                weekly_team_record[team] = {
                    "Record": f"{wins}-{losses}-{ties} ({win_percentage:.2f}%)",
                    "Result": week_result,
                }

            # Append the week's records to the list
            weekly_record_list.append((week, weekly_team_record))

        # Generate HTML for weekly records
        combined_tables_by_week = ""

        for week, team_records in weekly_record_list:
            # Convert the team's records for this week to a DataFrame
            team_record_df = pd.DataFrame([
                {"Team": team, "Record": record["Record"], "Result": record["Result"]}
                for team, record in team_records.items()
            ])

            # Add a heading for the week's results
            week_table_html = f"<h3>{week} Results</h3>" + team_record_df.to_html(
                classes='table table-striped', index=False)

            # Append the week's table to the combined tables string
            combined_tables_by_week += week_table_html

    else:
        combined_tables_by_week = "<p>Necessary columns not found in data.</p>"

    # Check if necessary columns exist
    if all(col in data.columns for col in ['team_name', 'week', 'total_points', 'points_against', 'result']):
        # Initialize a dictionary to store the count of weekly records and results for each team
        team_weekly_record_counts = {}

        # Get the list of unique teams
        teams = sorted(data['team_name'].unique())  # Sort teams alphabetically upfront

        # Loop through each team
        for team in teams:
            # Filter data for this team
            team_data = data[data['team_name'] == team]

            # Initialize a dictionary to store the count of records and results for this team
            weekly_record_counts = {}

            # Loop through each week
            for week in sorted(team_data['week'].unique()):
                # Get the team's data for this week
                team_week_data = team_data[team_data['week'] == week]

                # Fetch the result for the team
                week_result = team_week_data['result'].values[0] if not team_week_data.empty else "N/A"
                week_add = team_week_data['week'].values[0] if not team_week_data.empty else "N/A"

                # Initialize weekly win/loss counters
                wins = 0
                losses = 0
                ties = 0

                # List to store the actual results for this week
                weekly_results = [week_result]
                weekly_week = [week_add]

                # Compare this team's points against all opponents in the same week
                for opponent_team in data['team_name'].unique():
                    if opponent_team != team:
                        opponent_week_data = data[
                            (data['team_name'] == opponent_team) & (data['week'] == week)
                            ]

                        if not opponent_week_data.empty:
                            # Compare points and update weekly win/loss record
                            if team_week_data['total_points'].values[0] > opponent_week_data['total_points'].values[0]:
                                wins += 1
                            elif team_week_data['total_points'].values[0] < opponent_week_data['total_points'].values[
                                0]:
                                losses += 1
                            else:
                                ties += 1

                # Generate the record string for this week
                weekly_record = f"{wins}-{losses}-{ties}"

                # Update the count for this record and append the result
                if weekly_record in weekly_record_counts:
                    weekly_record_counts[weekly_record]['count'] += 1
                    weekly_record_counts[weekly_record]['results'].extend(weekly_results)
                    weekly_record_counts[weekly_record]['weeks'].extend(weekly_week)
                else:
                    weekly_record_counts[weekly_record] = {'count': 1, 'results': weekly_results, 'weeks': weekly_week}

            # Store the aggregated counts for this team
            team_weekly_record_counts[team] = weekly_record_counts

        # Generate HTML for the aggregated record counts and results
        combined_tables_by_team = ""

        for team in sorted(team_weekly_record_counts.keys(), key=lambda x: x.lower()):  # Sort by team name, case-insensitive
            # Convert record counts to a DataFrame
            record_counts_data = []
            for record, details in team_weekly_record_counts[team].items():
                # Join results for that record as a comma-separated string
                results_list = ', '.join(details['results'])  # Join the results list into a string
                weeks_list = ', '.join(details['weeks'])
                record_counts_data.append([record, details['count'], results_list, weeks_list])

            # Create DataFrame for the team record counts and results
            record_counts_df = pd.DataFrame(record_counts_data, columns=['Record', 'Count', 'Results', 'Weeks'])

            # Sort the DataFrame by Count in descending order
            record_counts_df = record_counts_df.sort_values(by='Count', ascending=False)

            # Generate a summary table for the team
            team_table_html = (
                    f"<h3>{team} - Weekly Record Counts</h3>"
                    + record_counts_df.to_html(classes='table table-striped', index=False)
            )

            # Append the team's table to the combined tables string
            combined_tables_by_team += team_table_html

    else:
        combined_tables_by_team = "<p>Necessary columns not found in data.</p>"

    # Ensure required columns exist
    if all(col in data.columns for col in ['team_name', 'opponent', 'week', 'total_points']):
        teams = data['team_name'].unique()  # Get unique team names
        team_tables = {}  # Dictionary to store tables for each team
        average_ranks = {}  # Dictionary to store average ranks for sorting

        # Loop through each team
        for team in teams:
            # Filter data for the selected team
            team_df = data[data['team_name'] == team].copy()

            # Rank the team's total points for the season
            team_df['score_rank'] = team_df['total_points'].rank(ascending=False, method='min').astype(int)

            # Add opponent's data for each week
            opponent_df = data[['team_name', 'week', 'total_points']].copy()
            opponent_df.rename(columns={'team_name': 'opponent', 'total_points': 'opponent_total_points'}, inplace=True)
            opponent_df['opponent_score_rank'] = opponent_df.groupby('opponent')['opponent_total_points'].rank(
                ascending=False, method='min').astype(int)

            # Merge opponent's data into the team's data
            team_df = team_df.merge(
                opponent_df,
                on=['opponent', 'week'],
                how='left'
            )

            # Select relevant columns for the final table
            final_df = team_df[['week', 'total_points', 'score_rank',
                                'opponent', 'opponent_total_points', 'opponent_score_rank']]

            # Convert week to numeric for sorting
            final_df['week_numeric'] = final_df['week'].str.extract('(\d+)').astype(int)

            # Sort the DataFrame by the numeric week
            final_df.sort_values(by='week_numeric', inplace=True)

            # Drop the temporary week_numeric column
            final_df.drop(columns=['week_numeric'], inplace=True)

            # Calculate the average opponent score rank and round it to one decimal
            avg_opponent_score_rank = round(final_df['opponent_score_rank'].mean(), 1)

            # Append the average row to final_df
            avg_row = pd.DataFrame({
                'week': ['Average'],
                'total_points': [''],  # or use np.nan for numeric columns
                'score_rank': [''],
                'opponent': [''],
                'opponent_total_points': [''],
                'opponent_score_rank': [avg_opponent_score_rank]
            })

            final_df = pd.concat([final_df, avg_row], ignore_index=True)

            # Store the HTML table and average rank in the dictionaries
            team_tables[team] = final_df.to_html(classes='table table-striped', index=False)
            average_ranks[team] = avg_opponent_score_rank

        # Sort teams based on the average opponent score rank
        sorted_teams = sorted(average_ranks, key=average_ranks.get)

        # Initialize combined tables for all teams
        combined_tables2 = ""

        # Loop through the sorted team names and append their tables to the combined output
        for rank, team in enumerate(sorted_teams, start=1):
            combined_tables2 += f"<h3>{team} (Rank: {rank})</h3>{team_tables[team]}<br>"

    else:
        combined_tables2 = "<p>Necessary columns not found in data.</p>"

    # Ensure required columns exist
    if all(col in data.columns for col in ['team_name', 'opponent', 'week', 'total_points', 'points_against']):
        combined_tables3 = ""
        teams = data['team_name'].unique()  # Get unique team names
        team_tables2 = {}  # Dictionary to store tables for each team

        # Loop through each team
        for team in teams:
            # Filter data for the selected team
            team_df = data[data['team_name'] == team].copy()
            team_df = team_df[['team_name', 'week', 'total_points']]

            # Initialize a list to store records for each opponent
            records = []

            # Loop through all opponents (including the current team)
            for opponent in teams:
                # Filter data for the current opponent
                opponent_df = data[data['team_name'] == opponent][
                    ['team_name', 'opponent', 'week', 'points_against', 'total_points']
                ].copy()

                # Rename columns to avoid confusion after merging
                opponent_df.rename(
                    columns={'team_name': 'schedule_being_played'},  # To differentiate team names
                    inplace=True
                )

                # Merge opponent's data into the team's data by week
                merged_df = team_df.merge(opponent_df, on='week', how='left')

                # Handle case where opponent is the same as the team
                merged_df['points_to_compare'] = merged_df.apply(
                    lambda row: row['total_points_y'] if row['opponent'] == team else row['points_against'], axis=1
                )

                # Drop rows where team_name and opponent are the same
                filtered_df = merged_df.copy()

                # Calculate wins and losses against this opponent
                wins = (filtered_df['total_points_x'] > filtered_df['points_to_compare']).sum()
                losses = (filtered_df['total_points_x'] < filtered_df['points_to_compare']).sum()
                ties = (filtered_df['total_points_x'] == filtered_df['points_to_compare']).sum()

                # Store the W-L record for this opponent
                records.append({'Schedule Being Played': opponent, 'Record': f"{wins}-{losses}-{ties}"})

            # Create a DataFrame for the summary records
            summary_df = pd.DataFrame(records)

            # Convert the summary DataFrame to an HTML table
            summary_table = summary_df.to_html(classes='table table-striped', index=False)

            # Store the HTML table for this team
            team_tables2[team] = summary_table

        # Sort and combine the HTML tables for all teams
        sorted_team_tables = dict(sorted(team_tables2.items(), key=lambda item: item[0].lower()))
        for team, table in sorted_team_tables.items():
            combined_tables3 += f"<h3>{team}</h3>{table}<br>"

    else:
        combined_tables3 = "<p>Necessary columns not found in data.</p>"

    # Check if necessary columns exist
    if all(col in data.columns for col in ['team_name', 'opponent', 'week', 'total_points', 'points_against']):
        combined_tables_by_team2 = ""
        teams = data['team_name'].unique()  # Get unique team names

        # Dictionary to store aggregated counts for each team
        team_record_counts = {}

        # Loop through each team
        for team in teams:
            # Filter data for the selected team
            team_df = data[data['team_name'] == team].copy()
            team_df = team_df[['team_name', 'week', 'total_points']]

            # Dictionary to store record counts for this team
            record_counts = {}

            # Loop through all opponents
            for opponent in teams:
                # Filter data for the current opponent
                opponent_df = data[data['team_name'] == opponent][
                    ['team_name', 'opponent', 'week', 'points_against', 'total_points']
                ].copy()

                # Rename columns to avoid confusion after merging
                opponent_df.rename(
                    columns={'team_name': 'schedule_being_played'},  # To differentiate team names
                    inplace=True
                )

                # Merge opponent's data into the team's data by week
                merged_df = team_df.merge(opponent_df, on='week', how='left')

                # Calculate points to compare
                merged_df['points_to_compare'] = merged_df.apply(
                    lambda row: row['total_points_y'] if row['opponent'] == team else row['points_against'], axis=1
                )

                # Calculate wins, losses, and ties for this opponent
                wins = (merged_df['total_points_x'] > merged_df['points_to_compare']).sum()
                losses = (merged_df['total_points_x'] < merged_df['points_to_compare']).sum()
                ties = (merged_df['total_points_x'] == merged_df['points_to_compare']).sum()

                # Generate the record string
                record = f"{wins}-{losses}-{ties}"

                # Update the record count
                if record in record_counts:
                    record_counts[record] += 1
                else:
                    record_counts[record] = 1

            # Store the aggregated record counts for this team
            team_record_counts[team] = record_counts

        # Generate HTML for the aggregated record counts, sorted alphabetically by team names (case-insensitive)
        for team in sorted(team_record_counts.keys(), key=lambda x: x.lower()):
            # Get the record counts for the current team
            record_counts = team_record_counts[team]

            # Convert record counts to a DataFrame
            record_counts_df = pd.DataFrame(
                [{'Record': record, 'Count': count} for record, count in record_counts.items()]
            )

            # Sort the DataFrame by Count in descending order
            record_counts_df = record_counts_df.sort_values(by='Count', ascending=False)

            # Generate a summary table for the team
            team_table_html = (
                    f"<h3>{team} - Record Counts</h3>"
                    + record_counts_df.to_html(classes='table table-striped', index=False)
            )

            # Append the team's table to the combined tables string
            combined_tables_by_team2 += team_table_html


    else:
        combined_tables_by_team2 = "<p>Necessary columns not found in data.</p>"

    # Prepare context with the HTML tables
    context = {
        'average_differential_table': average_differential_table,
        'average_by_team_table': average_by_team_table,
        'average_by_team_table_drop_two_lowest': average_by_team_table_drop_two_lowest,
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
        'max_difference_points_by_team_table': max_difference_points_by_team_table,
        'min_difference_points_by_team_table': min_difference_points_by_team_table,
        'min_points_by_team_table': min_points_by_team_table,
        'margins_counts_table': margins_counts_table,
        'record_against_everyone': combined_tables,
        'record_against_everyone_by_week': combined_tables_by_week,
        'record_against_everyone_by_team': combined_tables_by_team,
        'score_rank_by_team': combined_tables2,
        'different_teams_schedules': combined_tables3,
        'different_teams_schedules_count': combined_tables_by_team2,
        'years': years,
        'selected_year': selected_year,
        'page_title': f'Fantasy Stat Tables ({selected_year})'
    }

    # Render the HTML content using the context
    html_content = render_to_string('fantasy_data/stats.html', context)
    return HttpResponse(html_content)


def stats_charts_filter(request):
    # Get selected points category and comparison operator
    selected_category = request.GET.get('points_category')
    comparison_operator = request.GET.get('comparison_operator', 'gte')  # Default to 'gte' if not specified
    
    # Get available years and selected year
    years = TeamPerformance.objects.values_list('year', flat=True).distinct().order_by('year')
    selected_year = request.GET.get('year', years.last() if years else 2023)
    
    # Convert selected_year to integer
    try:
        selected_year = int(selected_year)
    except (ValueError, TypeError):
        selected_year = years.last() if years else 2023

    # Filter data based on user input and year
    filter_value = request.GET.get('filter_value')
    if filter_value:
        filter_value = float(filter_value)
        # Use the comparison operator chosen by the user
        filter_key = f"{selected_category}__{comparison_operator}"
        filtered_data = TeamPerformance.objects.filter(year=selected_year, **{filter_key: filter_value})
    else:
        filtered_data = TeamPerformance.objects.filter(year=selected_year)

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
        context = {
            'labels': labels, 
            'counts': counts, 
            'winCounts': win_counts, 
            'years': years,
            'selected_year': selected_year,
            'page_title': f'Fantasy Stats G/L Than ({selected_year})'
        }
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


def prepare_data(year=None):
    # Define required columns
    required_columns = [
        'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total',
        'k_points', 'def_points', 'total_points', 'result', 'team_name', 'opponent'
    ]
    
    # Only fetch the fields we need
    if year:
        team_performance = TeamPerformance.objects.filter(year=year).values(*required_columns)
    else:
        team_performance = TeamPerformance.objects.all().values(*required_columns)
        
    # Check if we have any data
    if not team_performance.exists():
        logger.warning("No data found for prepare_data")
        return pd.DataFrame(), None, None, None
        
    df = pd.DataFrame(list(team_performance))
    
    # Check if we still have data after dropping NA values
    if df.empty:
        logger.warning("Empty dataframe after initial load")
        return df, None, None, None

    # Ensure numerical columns are in numeric format
    numeric_columns = [
        'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total',
        'k_points', 'def_points', 'total_points'
    ]
    
    # Convert all numeric columns at once
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with NaNs in numeric columns
    df = df.dropna(subset=numeric_columns)
    
    # Check if we still have data after converting to numeric
    if df.empty:
        logger.warning("Empty dataframe after converting to numeric")
        return df, None, None, None

    # Encode categorical columns
    encoders = {}
    for col in ['team_name', 'opponent', 'result']:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col].astype(str))

    # Binary outcome: 1 for win, 0 for tie, -1 for loss
    # Use numpy vectorized operations instead of apply
    df['win'] = np.select(
        [df['result'] == 2, df['result'] == 0], 
        [1, 0], 
        default=-1
    )

    return df, encoders['team_name'], encoders['opponent'], encoders['result']


def train_model(year=None):
    # Use a cache key based on the year
    cache_key = f"fantasy_model_{year}" if year else "fantasy_model_all"
    
    # Try to get model from cache
    cached_model = getattr(train_model, 'cache', {}).get(cache_key)
    if cached_model:
        logger.info(f"Using cached model for year {year}")
        return cached_model
    
    df, label_encoder_team, label_encoder_opponent, label_encoder_result = prepare_data(year)
    
    # Check if we have valid data
    if df.empty or label_encoder_team is None:
        logger.warning(f"Invalid data for model training: year={year}")
        return None, None, None, None
        
    # Define required columns
    required_columns = ['qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total', 
                        'k_points', 'def_points', 'total_points', 'win']
                        
    # Check if all required columns exist
    if not all(col in df.columns for col in required_columns):
        logger.warning(f"Missing required columns for model training: year={year}")
        return None, None, None, None
    
    try:
        # Define features and target
        feature_columns = ['qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total', 
                          'k_points', 'def_points', 'total_points']
        X = df[feature_columns]
        y = df['win']
        
        # Check if we have enough data to train
        if len(X) < 2:
            logger.warning(f"Not enough data to train model: year={year}, samples={len(X)}")
            return None, label_encoder_team, label_encoder_opponent, label_encoder_result

        # Train the logistic regression model
        model = LogisticRegression(max_iter=1000)  # Increase max_iter for better convergence
        model.fit(X, y)

        # Calculate the overall model accuracy if we have enough data
        if len(X) >= 4:  # Need at least 4 samples to do a train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model Accuracy for year {year}: {accuracy:.2f}")
        
        # Cache the model
        if not hasattr(train_model, 'cache'):
            train_model.cache = {}
        train_model.cache[cache_key] = (model, label_encoder_team, label_encoder_opponent, label_encoder_result)
        
        return model, label_encoder_team, label_encoder_opponent, label_encoder_result
        
    except Exception as e:
        logger.error(f"Error training model for year {year}: {e}")
        return None, label_encoder_team, label_encoder_opponent, label_encoder_result


def versus(request):
    page_title = "Team Comparison"

    # Get available years and selected year
    years = TeamPerformance.objects.values_list('year', flat=True).distinct().order_by('year')
    selected_year = request.GET.get('year', years.last() if years else 2023)
    
    # Convert selected_year to integer
    try:
        selected_year = int(selected_year)
    except (ValueError, TypeError):
        selected_year = years.last() if years else 2023
    
    # Get teams for the selected year
    teams = TeamPerformance.objects.filter(year=selected_year).values_list('team_name', flat=True).distinct().order_by('team_name')
    team1 = request.GET.get('team1', None)
    team2 = request.GET.get('team2', None)

    if not team1 or not team2:
        return render(request, 'fantasy_data/versus.html', {
            'teams': teams, 
            'years': years,
            'selected_year': selected_year,
            'page_title': page_title
        })

    # Check if data exists for the selected teams and year
    team_performance = TeamPerformance.objects.filter(team_name__in=[team1, team2], year=selected_year)
    if not team_performance.exists():
        return render(request, 'fantasy_data/versus.html', {
            'teams': teams,
            'years': years,
            'selected_year': selected_year,
            'error': f"No data found for the selected teams in {selected_year}",
            'page_title': page_title
        })

    # Load and prepare data for prediction
    model, label_encoder_team, label_encoder_opponent, label_encoder_result = train_model(selected_year)

    df = pd.DataFrame(list(team_performance.values()))
    
    # Define numeric columns
    numeric_columns = [
        'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total',
        'k_points', 'def_points', 'total_points'
    ]
    
    # Check if all required columns exist in the dataframe
    if not all(col in df.columns for col in numeric_columns):
        return render(request, 'fantasy_data/versus.html', {
            'teams': teams,
            'years': years,
            'selected_year': selected_year,
            'error': f"Missing required data columns for the selected teams in {selected_year}",
            'page_title': page_title
        })
        
    # Convert columns to numeric and handle missing values
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Check if we have data for both teams
    if df[df['team_name'] == team1].empty or df[df['team_name'] == team2].empty:
        return render(request, 'fantasy_data/versus.html',
                      {'teams': teams,
                       'years': years,
                       'selected_year': selected_year,
                       'error': f"No data available for one or both teams in {selected_year}",
                       'page_title': page_title})

    # Check if teams are in the encoder classes
    try:
        if team1 not in label_encoder_team.classes_ or team2 not in label_encoder_team.classes_:
            return render(request, 'fantasy_data/versus.html',
                        {'teams': teams,
                        'years': years,
                        'selected_year': selected_year,
                        'error': f"One or both teams '{team1}' and '{team2}' not found in training data.",
                        'page_title': page_title})
    except (AttributeError, TypeError):
        # Handle case where label_encoder_team.classes_ doesn't exist or is None
        return render(request, 'fantasy_data/versus.html',
                    {'teams': teams,
                    'years': years,
                    'selected_year': selected_year,
                    'error': f"Unable to process team data for {selected_year}",
                    'page_title': page_title})

    # Prepare the feature vectors for the teams
    df_team1 = df[df['team_name'] == team1].mean(numeric_only=True)
    df_team2 = df[df['team_name'] == team2].mean(numeric_only=True)

    # Check if we have valid numeric data
    if df_team1.empty or df_team2.empty or not all(col in df_team1.index for col in numeric_columns) or not all(col in df_team2.index for col in numeric_columns):
        return render(request, 'fantasy_data/versus.html',
                    {'teams': teams,
                    'years': years,
                    'selected_year': selected_year,
                    'error': f"Insufficient data for comparison in {selected_year}",
                    'page_title': page_title})

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

    # Check if model is available
    if model is None:
        # If no model is available, use a simple comparison of average points
        prediction = "No prediction model available. Using simple point comparison instead."
        prob_team1_normalized = 0.5
        prob_team2_normalized = 0.5
        
        # Simple comparison based on average points
        if df_team1['total_points'] > df_team2['total_points']:
            prob_team1_normalized = 0.6
            prob_team2_normalized = 0.4
        elif df_team1['total_points'] < df_team2['total_points']:
            prob_team1_normalized = 0.4
            prob_team2_normalized = 0.6
    else:
        # Use the model for prediction
        try:
            # Predict probabilities for each team
            prob_team1_win = model.predict_proba(X_team1)[:, model.classes_ == 1][0][0]
            prob_team2_win = model.predict_proba(X_team2)[:, model.classes_ == 1][0][0]
            
            # Normalize win probabilities
            total_prob = prob_team1_win + prob_team2_win
            prob_team1_normalized = prob_team1_win / total_prob
            prob_team2_normalized = prob_team2_win / total_prob
        except (IndexError, ValueError) as e:
            # Fallback if prediction fails
            print(f"Prediction error: {e}")
            prob_team1_normalized = 0.5
            prob_team2_normalized = 0.5

    # Generate prediction messages including the chance of a tie
    if not 'prediction' in locals():  # Only generate if not already set in the model is None case
        if prob_team1_normalized > prob_team2_normalized:
            prediction = (
                f"{team1} is more likely to win with a probability of {prob_team1_normalized:.2%}. "
                f"{team2} has a probability of {prob_team2_normalized:.2%} to win. "
            )
        elif prob_team1_normalized < prob_team2_normalized:
            prediction = (
                f"{team2} is more likely to win with a probability of {prob_team2_normalized:.2%}. "
                f"{team1} has a probability of {prob_team1_normalized:.2%} to win. "
            )
        else:
            prediction = (
                "It's too close to call! Both teams have similar chances with "
                f"{team1} at {prob_team1_normalized:.2%} and {team2} at {prob_team2_normalized:.2%}. "
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
        'difference_between_teams': difference_between_teams,
        'years': years,
        'selected_year': selected_year,
        'page_title': f"{page_title} ({selected_year})"
    }

    return render(request, 'fantasy_data/versus.html', context)


def win_probability_against_all_teams(request):
    page_title = "Fantasy Win Probability"

    # Get available years and selected year
    years = TeamPerformance.objects.values_list('year', flat=True).distinct().order_by('year')
    selected_year = request.GET.get('year', years.last() if years else 2023)
    
    # Convert selected_year to integer
    try:
        selected_year = int(selected_year)
    except (ValueError, TypeError):
        selected_year = years.last() if years else 2023
    
    # Get teams for the selected year
    teams = TeamPerformance.objects.filter(year=selected_year).values_list('team_name', flat=True).distinct().order_by('team_name')
    selected_team = request.GET.get('team', None)

    if not selected_team:
        return render(request, 'fantasy_data/probabilities.html', {
            'teams': teams, 
            'years': years,
            'selected_year': selected_year,
            'page_title': page_title
        })

    # Check if data exists for the selected team and year
    team_data = TeamPerformance.objects.filter(team_name=selected_team, year=selected_year)
    if not team_data.exists():
        return render(request, 'fantasy_data/probabilities.html', {
            'teams': teams,
            'years': years,
            'selected_year': selected_year,
            'error': f"No data found for {selected_team} in {selected_year}",
            'page_title': page_title
        })

    # Load and prepare data for prediction
    model, label_encoder_team, label_encoder_opponent, label_encoder_result = train_model(selected_year)

    team_performance = TeamPerformance.objects.filter(year=selected_year)
    df = pd.DataFrame(list(team_performance.values()))

    # Define numeric columns
    numeric_columns = [
        'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total',
        'k_points', 'def_points', 'total_points'
    ]
    
    # Check if all required columns exist in the dataframe
    if not all(col in df.columns for col in numeric_columns):
        return render(request, 'fantasy_data/probabilities.html', {
            'teams': teams,
            'years': years,
            'selected_year': selected_year,
            'error': f"Missing required data columns for {selected_year}",
            'page_title': page_title
        })
        
    # Convert columns to numeric and handle missing values
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
    # Drop rows with NaNs in numeric columns
    df = df.dropna(subset=numeric_columns)
    
    # Check if we have data after cleaning
    if df.empty:
        return render(request, 'fantasy_data/probabilities.html', {
            'teams': teams,
            'years': years,
            'selected_year': selected_year,
            'error': f"No valid data found after cleaning for {selected_year}",
            'page_title': page_title
        })

    # Check if model and encoder are available
    if model is None or label_encoder_team is None:
        return render(request, 'fantasy_data/probabilities.html', {
            'teams': teams,
            'years': years,
            'selected_year': selected_year,
            'error': f"Unable to create prediction model for {selected_year}",
            'page_title': page_title
        })

    # Check if selected team is in encoder classes
    try:
        if selected_team not in label_encoder_team.classes_:
            return render(request, 'fantasy_data/probabilities.html', {
                'teams': teams,
                'years': years,
                'selected_year': selected_year,
                'error': f"Team '{selected_team}' not found in training data.",
                'page_title': page_title
            })
    except (AttributeError, TypeError):
        return render(request, 'fantasy_data/probabilities.html', {
            'teams': teams,
            'years': years,
            'selected_year': selected_year,
            'error': f"Error processing team data for {selected_year}",
            'page_title': page_title
        })

    # Prepare the feature vector for the selected team
    df_selected_team = df[df['team_name'] == selected_team].mean(numeric_only=True)
    
    # Check if we have data for the selected team
    if df_selected_team.empty:
        return render(request, 'fantasy_data/probabilities.html', {
            'teams': teams,
            'years': years,
            'selected_year': selected_year,
            'error': f"No data available for {selected_team} in {selected_year}",
            'page_title': page_title
        })

    win_probabilities = []
    for opponent in teams:
        if opponent == selected_team:
            continue

        # Check if opponent has data
        opponent_data = df[df['team_name'] == opponent]
        if opponent_data.empty:
            continue

        # Prepare the feature vector for the opponent
        df_opponent = opponent_data.mean(numeric_only=True)

        # Calculate the absolute difference between the two feature vectors
        feature_differences = df_selected_team[numeric_columns] - df_opponent[numeric_columns]

        # Prepare the feature vectors for prediction
        X_selected_team = pd.DataFrame([df_selected_team[numeric_columns].values], columns=numeric_columns)
        X_opponent = pd.DataFrame([df_opponent[numeric_columns].values], columns=numeric_columns)

        try:
            # Predict the probability of winning for each team
            prob_selected_team = model.predict_proba(X_selected_team)[:, model.classes_ == 1][0][0]
            prob_opponent = model.predict_proba(X_opponent)[:, model.classes_ == 1][0][0]

            # Normalize probabilities
            total_prob = prob_selected_team + prob_opponent
            prob_selected_team_normalized = prob_selected_team / total_prob
            prob_opponent_normalized = prob_opponent / total_prob
        except (IndexError, ValueError):
            # Fallback to simple comparison if prediction fails
            if df_selected_team['total_points'] > df_opponent['total_points']:
                prob_selected_team_normalized = 0.6
                prob_opponent_normalized = 0.4
            elif df_selected_team['total_points'] < df_opponent['total_points']:
                prob_selected_team_normalized = 0.4
                prob_opponent_normalized = 0.6
            else:
                prob_selected_team_normalized = 0.5
                prob_opponent_normalized = 0.5

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
        'years': years,
        'selected_year': selected_year,
        'page_title': f"{page_title} ({selected_year})"
    })


def top_tens(request):
    # Get available years and selected year
    years = TeamPerformance.objects.values_list('year', flat=True).distinct().order_by('year')
    selected_year = request.GET.get('year', years.last() if years else 2023)
    
    # Convert selected_year to integer
    try:
        selected_year = int(selected_year)
    except (ValueError, TypeError):
        selected_year = years.last() if years else 2023
        
    # Retrieve data from the database filtered by year
    data = TeamPerformance.objects.filter(year=selected_year).values()

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
    df = pd.DataFrame(data)

    # Filtering for smallest win margins where the result is 'W'
    win_margin_df = df[(df['margin'] > 0) & (df['result'] == 'W')]

    # Find the top 10 smallest win margins by week
    smallest_win_margins_rows = win_margin_df.groupby('week')['margin'].nsmallest(10).index.get_level_values(1)
    smallest_win_margins = win_margin_df.loc[
        smallest_win_margins_rows, ['week', 'team_name', 'total_points', 'margin', 'opponent']]
    smallest_win_margins = smallest_win_margins.sort_values(by=['margin'], ascending=True).reset_index(drop=True)
    smallest_win_margins = smallest_win_margins.head(10)  # Limit to top 10 rows
    smallest_win_margins.index = smallest_win_margins.index + 1

    # Get the top 10 largest margins by week
    largest_win_margins_rows = df.groupby('week')['margin'].nlargest(10).index.get_level_values(1)
    largest_win_margins = df.loc[largest_win_margins_rows, ['week', 'team_name', 'total_points', 'margin', 'opponent']]
    largest_win_margins = largest_win_margins.sort_values(by=['margin'], ascending=False).reset_index(drop=True)
    largest_win_margins = largest_win_margins.head(10)  # Limit to top 10 rows
    largest_win_margins.index = largest_win_margins.index + 1

    # Get the top 10 largest projection margins by week
    largest_win_projection_rows = df.groupby('week')['difference'].nlargest(10).index.get_level_values(1)
    largest_win_projections = df.loc[largest_win_projection_rows, ['week', 'team_name', 'total_points',
                                                                   'expected_total', 'difference', 'opponent',
                                                                   'result']]
    largest_win_projections = largest_win_projections.sort_values(by=['difference'], ascending=False).reset_index(
        drop=True)
    largest_win_projections = largest_win_projections.head(10)  # Limit to top 10 rows
    largest_win_projections.index = largest_win_projections.index + 1

    # Get the top 10 largest negative projection margins by week
    smallest_win_projection_rows = df.groupby('week')['difference'].nsmallest(10).index.get_level_values(1)
    smallest_win_projections = df.loc[smallest_win_projection_rows, ['week', 'team_name', 'total_points',
                                                                     'expected_total', 'difference', 'opponent',
                                                                     'result']]
    # Sort by 'difference' in ascending order to keep the largest negative values at the top
    smallest_win_projections = smallest_win_projections.sort_values(by=['difference'], ascending=True).reset_index(
        drop=True)
    # Limit to top 10 rows
    smallest_win_projections = smallest_win_projections.head(10)
    smallest_win_projections.index = smallest_win_projections.index + 1

    # Repeat for each position: QB, WR, RB, TE, K, DEF, for both largest and smallest values
    def get_top_10_by_position(position, df, largest=True):
        group_method = 'nlargest' if largest else 'nsmallest'
        top_rows = df.groupby('week')[position].agg(group_method, 10).index.get_level_values(1)
        result = df.loc[top_rows, ['week', 'team_name', position, 'opponent', 'result']]
        order = False if largest else True
        result = result.sort_values(by=[position], ascending=order).reset_index(drop=True)
        result = result.head(10)  # Limit to top 10 rows
        result.index = result.index + 1
        return result

    positions = ['total_points', 'qb_points', 'wr_points_total', 'rb_points_total', 'te_points_total', 'k_points',
                 'def_points']

    largest_positions_tables = {
        f'top_10_{position}_largest_table': get_top_10_by_position(position, df, largest=True).to_html(
            classes='table table-striped') for position in positions}
    smallest_positions_tables = {
        f'top_10_{position}_smallest_table': get_top_10_by_position(position, df, largest=False).to_html(
            classes='table table-striped') for position in positions}

    # Convert the DataFrames to HTML
    largest_win_margins_table = largest_win_margins.to_html(classes='table table-striped')
    smallest_win_margins_table = smallest_win_margins.to_html(classes='table table-striped')
    largest_win_projections = largest_win_projections.to_html(classes='table table-striped')
    smallest_win_projections = smallest_win_projections.to_html(classes='table table-striped')

    # Prepare context for rendering in the template
    context = {
        'top_10_largest_win_margins_table': largest_win_margins_table,
        'top_10_smallest_win_margins_table': smallest_win_margins_table,
        'top_10_largest_win_projections_table': largest_win_projections,
        'top_10_smallest_win_projections_table': smallest_win_projections,
        **largest_positions_tables,
        **smallest_positions_tables,
        'years': years,
        'selected_year': selected_year,
        'page_title': f'Top 10s Tables ({selected_year})'
    }

    # Render the results in the top_tens.html template
    return render(request, 'fantasy_data/top_tens.html', context)
