import pandas as pd
import os
from datetime import datetime as dt
from datetime import timedelta, date
import plotly.graph_objs as go
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
from app import app


def alpha_date(date):
    """
    :param date: MM/DD/YYYY
    :return: YYYY_MM_DD
    """
    date = date.replace("/", "_")
    alphadate = date[6:] + "_" + date[:5]
    return alphadate


# Style dictionaries for dashboard elements:
input_dates_style = {'fontFamily': 'helvetica', 'fontSize': 12, 'display': 'block'}
dropdown_div_style = {'width': '100%', 'float': 'left', 'display': 'block'}
graph_style = {'width': '58%', 'display': 'block', 'float': 'left'}
stats_style = {'width': '38%', 'display': 'block', 'float': 'left'}

# Defaults:
default_start_date = '2022-01-01'
default_end_date = dt.strftime(date.today(), "%Y-%m-%d")
default_gender_choice = 'women'
date_display_format = 'Y-M-D'
progressions_app_description = "Select multiple athletes to see both the their ranking and rating progressions " \
                               "over time on the same graph. The rating value is an arbitrary number produced by " \
                               "the ranking algorithm that is used to create the ranking. This tool allows a user " \
                               "to see how close, or far, an athlete is from surpassing or being surpassed by other " \
                               "athletes in the ranking pool."

layout = html.Div([
    html.Div(children=progressions_app_description),
    dcc.RadioItems(id='progressions-gender-picker',
                   options=[{'label': 'Men', 'value': 'men'}, {'label': 'Women', 'value': 'women'}],
                   persistence=True, persistence_type='session'),
    html.Div(dcc.Dropdown(id='progressions-name-dropdown',
                          multi=True, persistence=True, persistence_type='session'),
                          style=dropdown_div_style),
    html.Div([
            html.Label('Start Date '),
            dcc.DatePickerSingle(id='progressions-start-date', date=default_start_date, display_format=date_display_format,
                                    clearable=False, persistence=True, persistence_type='session'),
            html.Label('End Date '),
            dcc.DatePickerSingle(id='progressions-end-date', date=default_end_date, display_format=date_display_format,
                                    clearable=False, persistence=True, persistence_type='session'),
            ], style=input_dates_style),
    dcc.Graph(id='rank-progression-graph'),
    dcc.Graph(id='rating-progression-graph')
])

# Create / update ranking progression graph:
@app.callback(
    [Output('rank-progression-graph', 'figure'),
     Output('rating-progression-graph', 'figure')],
    [Input('progressions-start-date', 'date'),
     Input('progressions-end-date', 'date'),
     Input('progressions-name-dropdown', 'value'),
     Input('progressions-gender-picker', 'value')])
def ranking_progression(start_date, end_date, athlete_names, gender_choice):
    rating_multiplier = 10000
    athlete_names = list(athlete_names)
    start_date = dt.strptime(start_date, "%Y-%m-%d")
    end_date = dt.strptime(end_date, "%Y-%m-%d")
    increment = 1
    rank_dist = 10
    rankings_directory = 'app_data/' + gender_choice + "/rankings_archive"
    results_directory = 'app_data/' + gender_choice + "/results"
    athlete_data_directory = 'app_data/' + gender_choice + "/athlete_data"
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    rank_traces = []
    rating_traces = []
    scatter_df = pd.DataFrame()

    # Loop through each athlete and grab their progression csv. Create a dataframe, add a datetime format date column,
    # and filter it to include only rows in the selected date range. Then make a ranking and rating trace for them.
    # If no csv available, calculate the old way by looping through dated ranking files.
    for athlete_name in athlete_names:
        athlete_name = athlete_name.title()
        if os.path.exists(f"{athlete_data_directory}/{athlete_name}.csv"):
            df = pd.read_csv(f"{athlete_data_directory}/{athlete_name}.csv")
            df['dt_date'] = [dt.strptime(d, "%m/%d/%Y") for d in df['date']]
            df = df[df['dt_date'] >= start_date]
            df = df[df['dt_date'] <= end_date]

            # Use the columns of the modified dataframe to construct a trace for the rank line and the rating line.
            # Add the traces to the list that gets passed into the figure data parameter and returned to the graph.

            rank_line_trace = go.Scatter(x=df['dt_date'],
                                    y=df['rank'],
                                    mode='lines',
                                    opacity=0.8,
                                    name=athlete_name)
            rank_traces.append(rank_line_trace)

            rating_line_trace = go.Scatter(x=df['dt_date'],
                                         y=[i * rating_multiplier for i in df['rating']],
                                         mode='lines',
                                         opacity=0.8,
                                         name=athlete_name)
            rating_traces.append(rating_line_trace)
        else:
            rank_dates = []
            ranks = []
            ratings = []

            # For each date in the date range, look up the athlete's rank and rating on that date and create a trace for each.
            # Then add those traces to the respective lists to be used in the data parameter of each of the figures
            for date in date_range:
                file_name = f"{alpha_date(date)}_{gender_choice}_{rank_dist}km.csv"
                ranking_data = pd.read_csv(f"{rankings_directory}/{file_name}")
                ranked_athletes = list(ranking_data.name)
                if athlete_name in ranked_athletes:
                    rank_dates.append(dt.strptime(date, "%m/%d/%Y"))
                    rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
                    rating_on_date = float(ranking_data["pagerank"][ranking_data.name == athlete_name])
                    ranks.append(rank_on_date)
                    ratings.append(rating_on_date)
            ratings = [i * rating_multiplier for i in ratings]
            rank_line_trace = go.Scatter(x=rank_dates,
                                         y=ranks,
                                         mode='lines',
                                         opacity=0.8,
                                         name=athlete_name)
            rank_traces.append(rank_line_trace)
            rating_line_trace = go.Scatter(x=rank_dates,
                                           y=ratings,
                                           mode='lines',
                                           opacity=0.8,
                                           name=athlete_name)
            rating_traces.append(rating_line_trace)

        # Create the scatter traces.
        # this block is the get_results function from main.py basically. Get a df of all the times the athlete raced.
        rows = []
        for file in os.listdir(results_directory):
            results_file_path = os.path.join(results_directory, file)
            race_data = pd.read_csv(results_file_path)
            names_list = list(race_data.athlete_name)
            names_list = [name.title() for name in names_list]
            race_data.athlete_name = names_list
            if athlete_name.title() in names_list:
                row = race_data[race_data.athlete_name == athlete_name.title()]
                rows.append(row)
        results_df = pd.concat(rows, ignore_index=True)

        # add dt_date column and filter down to dates within selection
        results_df["dt_date"] = [dt.strptime(date, "%m/%d/%Y") for date in results_df.date]
        results_df = results_df[results_df.dt_date >= start_date]
        results_df = results_df[results_df.dt_date <= end_date]
        race_date_ranks = []
        race_date_ratings = []
        # for dates where athlete raced, look up their rank on that date and create another column for rank
        # do the same for their rating - now you have two new columns in the df.
        for date in results_df.date:
            file_name = f"{alpha_date(date)}_{gender_choice}_{rank_dist}km.csv"
            df = pd.read_csv(f"{rankings_directory}/{file_name}")
            rank_on_date = int(df["rank"][df.name == athlete_name])
            rating_on_date = float(df["pagerank"][df.name == athlete_name])
            race_date_ranks.append(rank_on_date)
            race_date_ratings.append(rating_on_date)
        results_df['rank'] = race_date_ranks
        results_df['rating'] = race_date_ratings
        scatter_df = pd.concat([scatter_df, results_df])

    for event_type in scatter_df['event'].unique():
        df = scatter_df[scatter_df['event'] == event_type]
        scatter_trace_rank = go.Scatter(x=df['dt_date'],
                                   y=df['rank'],
                                   mode='markers',
                                   marker={'size': 10, 'line': {'width': 0.5, 'color': 'black'}},
                                   name=event_type)
        rank_traces.append(scatter_trace_rank)
        scatter_trace_rating = go.Scatter(x=df['dt_date'],
                                        y=[i * rating_multiplier for i in df['rating']],
                                        mode='markers',
                                        marker={'size': 10, 'line': {'width': 0.5, 'color': 'black'}},
                                        name=event_type)
        rating_traces.append(scatter_trace_rating)

    rank_fig = {
        'data': rank_traces,
        'layout': go.Layout(
            xaxis={'title': 'Date'},
            yaxis={'title': 'World Ranking', 'autorange': 'reversed'},
            hovermode='closest')
    }

    rating_fig = {
        'data': rating_traces,
        'layout': go.Layout(
            xaxis={'title': 'Date'},
            yaxis={'title': 'Rating Value'},
            hovermode='closest')
    }

    return rank_fig, rating_fig


# Update names in dropdown list when a different gender is selected:
@app.callback(Output('progressions-name-dropdown', 'options'),
              [Input('progressions-gender-picker', 'value')])
def list_names(gender_choice):
    df = pd.read_csv('app_data/' + gender_choice + "/athlete_countries.csv")
    names = df['athlete_name'].unique()
    return [{'label': i, 'value': i} for i in names]
