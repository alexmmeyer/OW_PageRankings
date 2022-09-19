import pandas as pd
import os
from datetime import datetime as dt
from datetime import timedelta
import plotly.graph_objs as go
import plotly.offline as pyo
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output


def alpha_date(date):
    """
    :param date: MM/DD/YYYY
    :return: YYYY_MM_DD
    """
    date = date.replace("/", "_")
    alphadate = date[6:] + "_" + date[:5]
    return alphadate


# Style dictionaries for dashboard elements:
input_dates_style = {'fontFamily': 'helvetica', 'fontSize': 18, 'display': 'block'}
dropdown_div_style = {'width': '30%', 'float': 'left', 'height': 50, 'display': 'block'}
graph_style = {'width': '30%', 'display': 'block'}

app = Dash()

app.layout = html.Div([
    dcc.RadioItems(id='gender-picker', options=[{'label': 'Men', 'value': 'men'}, {'label': 'Women', 'value': 'women'}], value='women'),
    dcc.Input(id='start-date', value='01/01/2022'),
    dcc.Input(id='end-date', value='08/31/2022'),
    html.Div(dcc.Dropdown(id='name-dropdown', value='Lea Boy', options=[{'label': i, 'value': i} for i in pd.read_csv("women/athlete_countries.csv")]),
             style=dropdown_div_style),
    html.Div(id='output-text', children='starter text'),
    dcc.Graph(id='progression-graph'),
    html.Div(id='results-table', children='hello world')
])


@app.callback(
    Output('progression-graph', 'figure'),
    [Input('start-date', 'value'),
     Input('end-date', 'value'),
     Input('name-dropdown', 'value'),
     Input('gender-picker', 'value')])
def ranking_progression(start_date, end_date, athlete_name, gender_choice):
    athlete_names = [athlete_name]
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    increment = 1
    rank_dist = 10
    rankings_directory = gender_choice + "/rankings_archive"
    results_directory = gender_choice + "/results"
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    traces = []
    scatter_df = pd.DataFrame()

    for athlete_name in athlete_names:
        athlete_name = athlete_name.title()
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
        line_trace = go.Scatter(x=rank_dates,
                                y=ranks,
                                mode='lines',
                                opacity=0.8,
                                name=athlete_name)
        traces.append(line_trace)

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

        results_df["dt_date"] = [dt.strptime(date, "%m/%d/%Y") for date in results_df.date]
        results_df = results_df[results_df.dt_date >= start_date]
        results_df = results_df[results_df.dt_date <= end_date]
        race_date_ranks = []
        for date in results_df.date:
            file_name = f"{alpha_date(date)}_{gender_choice}_{rank_dist}km.csv"
            ranking_data = pd.read_csv(f"{rankings_directory}/{file_name}")
            rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
            race_date_ranks.append(rank_on_date)
        results_df['rank'] = race_date_ranks
        scatter_df = pd.concat([scatter_df, results_df])

    for event_type in scatter_df['event'].unique():
        df = scatter_df[scatter_df['event'] == event_type]
        scatter_trace = go.Scatter(x=df['dt_date'],
                                   y=df['rank'],
                                   mode='markers',
                                   marker={'size': 10, 'line': {'width': 0.5, 'color': 'black'}},
                                   name=event_type)
        traces.append(scatter_trace)

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'Date'},
            yaxis={'title': 'World Ranking', 'autorange': 'reversed'},
            hovermode='closest')
    }


# Update names in dropdown list when a different gender is selected:
@app.callback(Output('name-dropdown', 'options'),
              [Input('gender-picker', 'value')])
def list_names(gender_choice):
    df = pd.read_csv(gender_choice + "/athlete_countries.csv")
    names = df['athlete_name'].unique()
    return [{'label': i, 'value': i} for i in names]


# Display results in data table:
@app.callback(Output('results-table', 'children'),
              [Input('name-dropdown', 'value'),
               Input('gender-picker', 'value')])
def display_results(athlete_name, gender_choice):
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
    print(results_df)
    return athlete_name
    # return dash_table.DataTable(data=data, columns=columns)


if __name__ == '__main__':
    app.run_server()
