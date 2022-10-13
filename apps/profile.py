import pandas as pd
import os
from datetime import datetime as dt
from datetime import timedelta, date
import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
from app import app

today = dt.strftime(date.today(), "%m/%d/%Y")

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
summary_style = {'width': '100%', 'float': 'left', 'display': 'block'}
outcome_stats_style = {'width': '38%', 'display': 'block', 'float': 'left'}

default_start_date = '01/01/2022'
default_end_date = '09/30/2022'
default_gender = 'women'
default_athlete = 'Lea Boy'
default_names_list = pd.read_csv(f"{default_gender}/athlete_countries.csv").sort_values('athlete_name')

name_style = {'fontFamily': 'helvetica', 'fontSize': 72, 'textAlign': 'left'}
summary_stats_style = {'fontFamily': 'helvetica', 'fontSize': 36, 'textAlign': 'left'}

layout = html.Div([
    dcc.RadioItems(id='gender-picker', options=[{'label': 'Men', 'value': 'men'}, {'label': 'Women', 'value': 'women'}], value=default_gender),
    html.Div(dcc.Dropdown(id='name-dropdown', value=default_athlete,
                          options=[{'label': i, 'value': i} for i in default_names_list]),
                          style=dropdown_div_style),
    html.Div([
        html.H1(id='athlete-name', style=name_style),
        html.Div(id='summary-stats', style=summary_stats_style)
                ], style=summary_style),
    html.Div([
            html.Label('Start Date (MM/DD/YYYY)'),
            dcc.Input(id='start-date', value=default_start_date),
            html.Label('End Date (MM/DD/YYYY)'),
            dcc.Input(id='end-date', value=default_end_date),
            ], style=input_dates_style),
    dcc.Graph(id='progression-graph'),
    html.Div(id='outcome-stats-table', style=outcome_stats_style),
    html.Div(id='results-table')
])

# Create / update ranking progression graph:
@app.callback(
    Output('progression-graph', 'figure'),
    [Input('start-date', 'value'),
     Input('end-date', 'value'),
     Input('name-dropdown', 'value'),
     Input('gender-picker', 'value')])
def ranking_progression(start_date, end_date, athlete_name, gender_choice):
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    global today
    increment = 1
    rank_dist = 10
    rankings_directory = gender_choice + "/rankings_archive"
    results_directory = gender_choice + "/results"
    athlete_data_directory = gender_choice + "/athlete_data"
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    athlete_name = athlete_name.title()

    # If the athlete's data is precalculated, use their file to get the data for the progression line:
    if os.path.exists(f"{athlete_data_directory}/{athlete_name}.csv"):
        # Grab athlete's progression csv. Create a dataframe, add a datetime format date column, and filter it
        # to include only rows in the selected date range.
        df = pd.read_csv(f"{athlete_data_directory}/{athlete_name}.csv")
        current_wr = df['rank'][df['date'] == today]
        df['date'] = [dt.strptime(d, "%m/%d/%Y") for d in df['date']]
        df = df[df['date'] >= start_date]
        df = df[df['date'] <= end_date]

        # Use the columns of the modified dataframe to construct a px line plot.
        ln_fig = px.line(df, x='date', y='rank')

    # Otherwise loop through the ranking files in the date range and look up the athlete's ranking on that date.
    # Put the dates and ranks in two separate lists, then a df, and use those as x and y for the progression line trace.
    else:
        rank_dates = []
        ranks = []

        for date in date_range:
            file_name = f"{alpha_date(date)}_{gender_choice}_{rank_dist}km.csv"
            ranking_data = pd.read_csv(f"{rankings_directory}/{file_name}")
            ranked_athletes = list(ranking_data.name)
            if athlete_name in ranked_athletes:
                rank_dates.append(dt.strptime(date, "%m/%d/%Y"))
                rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
                ranks.append(rank_on_date)

        df = pd.DataFrame({
            'date': rank_dates,
            'rank': ranks
        })

        # Use the columns of the modified dataframe to construct a px line plot.
        ln_fig = px.line(df, x='date', y='rank')

    # Create the scatter trace:
    # this block is the get_results function
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

    # Add a datetime column and filter out the rows that are outside the selected time frame.
    results_df["dt_date"] = [dt.strptime(date, "%m/%d/%Y") for date in results_df.date]
    results_df = results_df[results_df.dt_date >= start_date]
    results_df = results_df[results_df.dt_date <= end_date]
    race_date_ranks = []
    # get the athlete's ranking on the date of each of their races in selected time frame and put it in a list
    for date in results_df.date:
        file_name = f"{alpha_date(date)}_{gender_choice}_{rank_dist}km.csv"
        ranking_data = pd.read_csv(f"{rankings_directory}/{file_name}")
        rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
        race_date_ranks.append(rank_on_date)
    # Then use that list to add a df column for rank (on race day).
    results_df['rank'] = race_date_ranks
    results_df["date"] = results_df["dt_date"]


    sp_fig = px.scatter(results_df, x="date", y="rank", color="event", hover_name="event",
                        hover_data=["location", "place", "field_size", "distance"])
    sp_fig.update_traces(marker=dict(size=12,
                                     line=dict(width=1.5,
                                               color='black')))

    layout = go.Layout(
        xaxis={'title': 'Date'},
        yaxis={'title': 'World Ranking', 'autorange': 'reversed'},
        hovermode='closest')

    fig = go.Figure(data=ln_fig.data + sp_fig.data, layout=layout)

    return fig

# Update names in dropdown list when a different gender is selected:
@app.callback(Output('name-dropdown', 'options'),
              [Input('gender-picker', 'value')])
def list_names(gender_choice):
    df = pd.read_csv(gender_choice + "/athlete_countries.csv")
    names = df.sort_values('athlete_name')
    names = names['athlete_name'].unique()
    return [{'label': i, 'value': i} for i in names]


# Update summary and outcomes stats tables for new athlete selected:
@app.callback([Output('summary-stats', 'children'),
               Output('outcome-stats-table', 'children'),
               Output('athlete-name', 'children')],
              [Input('name-dropdown', 'value'),
               Input('gender-picker', 'value')
               ])
def stats_table(athlete_name, gender_choice):

    # OUTCOMES SUMMARY
    wins = 0
    podiums = 0
    top25pct = 0
    total_races = 0
    dist = 'all'
    results_directory = gender_choice + "/results"
    athlete_countries = pd.read_csv(gender_choice + "/athlete_countries.csv")
    country = list(athlete_countries['country'][athlete_countries['athlete_name'] == athlete_name])[0]
    header = f"{athlete_name} ({country})"
    print(header)

    # loop through the results directory and if the athlete is in the results figure out what tier they ended up in in
    # that race and add to the corresponding list.
    for file in os.listdir(results_directory):
        results_file_path = os.path.join(results_directory, file)
        race_data = pd.read_csv(results_file_path)
        if athlete_name in list(race_data.athlete_name):
            race_dist = race_data.distance[0]
            finishers = len(race_data.athlete_name)
            if race_dist == dist or dist == "all":
                total_races += 1
                athlete_place = int(race_data.place[race_data.athlete_name == athlete_name])
                if athlete_place == 1:
                    wins += 1
                if athlete_place <= 3:
                    podiums += 1
                if athlete_place / finishers <= 0.25:
                    top25pct += 1

    tiers = [wins, podiums, top25pct]

    outcome_tiers = {
        'Outcome': ['Win', 'Podium', 'Top 25%'],
        'Count (all time)': [str(tier) + ' / ' + str(total_races) for tier in tiers],
        'Percentage': ["{:.0%}".format(num / total_races) for num in tiers]
    }

    outcomes_df = pd.DataFrame(outcome_tiers)
    data = outcomes_df.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in outcomes_df.columns]
    stats_datatable = [dash_table.DataTable(data=data, columns=columns)]

    # ATHLETE SUMMARY
    athlete_data_directory = gender_choice + "/athlete_data"
    if os.path.exists(f"{athlete_data_directory}/{athlete_name}.csv"):
        outcomes_df = pd.read_csv(f"{athlete_data_directory}/{athlete_name}.csv")
        current_wr = list(outcomes_df['rank'][outcomes_df['date'] == today])[0]
        highest_wr = min(outcomes_df['rank'])
        first_race_date = dt.strftime(min([dt.strptime(date, "%m/%d/%Y") for date in outcomes_df['date']]),"%m/%d/%Y")

        summary_text = html.P([f"Current WR: {current_wr}", html.Br(), f"Highest WR: {highest_wr}", html.Br(), f"Active Since: {first_race_date}"])

    # print(f"current wr: {current_wr}")
    # print(f"highest wr: {highest_wr}")
    # print(f"first race: {first_race_date}")

    return summary_text, stats_datatable, header


# Display results in data table for new athlete selected:
@app.callback(Output('results-table', 'children'),
              [Input('name-dropdown', 'value'),
               Input('gender-picker', 'value'),
               Input('start-date', 'value'),
               Input('end-date', 'value')
               ])
def results_table(athlete_name, gender_choice, start_date, end_date):
    print('display results callback triggering')
    results_directory = gender_choice + "/results"
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
    results_df["dt_date"] = [dt.strptime(date, "%m/%d/%Y") for date in results_df.date]
    # results_df = results_df[results_df.dt_date >= start_date]
    # results_df = results_df[results_df.dt_date <= end_date]
    data = results_df.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in results_df.columns]
    table = dash_table.DataTable(data=data, columns=columns)
    return [table]
