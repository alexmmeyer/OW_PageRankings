import pandas as pd
import os
import math
from datetime import datetime as dt
from datetime import timedelta
import plotly.graph_objs as go
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output


app = Dash()

stats_style = {'fontFamily': 'helvetica', 'fontSize': 48, 'float': 'center'}

dropdown_div_style = {'width': '50%', 'float': 'left', 'display': 'block'}

app.layout = html.Div([
    dcc.RadioItems(id='gender-picker', options=[{'label': 'Men', 'value': 'men'}, {'label': 'Women', 'value': 'women'}], value='women'),
    html.Div([
        html.Div(dcc.Dropdown(id='name-dropdown1', value='Lea Boy',
                     options=[{'label': i, 'value': i} for i in pd.read_csv("women/athlete_countries.csv")]), style=dropdown_div_style),
        html.Div(dcc.Dropdown(id='name-dropdown2', value='Caroline Laure Jouisse',
                     options=[{'label': i, 'value': i} for i in pd.read_csv("women/athlete_countries.csv")]), style=dropdown_div_style)]),
        html.H1(id='score', children='12 - 8'),
        html.Div(id='table')
    ])


# Update names in dropdown list 1 when a different gender is selected:
@app.callback([Output('name-dropdown1', 'options'),
               Output('name-dropdown2', 'options')],
              [Input('gender-picker', 'value')])
def list_names(gender_choice):
    df = pd.read_csv(gender_choice + "/athlete_countries.csv")
    names = df['athlete_name'].unique()
    names_list = [{'label': i, 'value': i} for i in names]
    return names_list, names_list


@app.callback([Output('table', 'children'),
               Output('score', 'children')],
              [Input('name-dropdown1', 'value'),
               Input('name-dropdown2', 'value'),
               Input('gender-picker', 'value')])
def update(name1, name2, gender_choice):
    dist = 'all'
    results_directory = gender_choice + "/results"
    winners = []
    winner_places = []
    loser_places = []
    table_diffs = []
    races = []

    chart_diffs = []

    for file in os.listdir(results_directory):
        results_file_path = os.path.join(results_directory, file)
        race_data = pd.read_csv(results_file_path)
        race_dist = race_data.distance[0]
        if name1 in list(race_data.athlete_name) and name2 in list(race_data.athlete_name):
            if race_dist == dist or dist == "all":
                races.append(file)
                name1time = float(race_data["time"][race_data["athlete_name"] == name1])
                name2time = float(race_data["time"][race_data["athlete_name"] == name2])
                diff = round(name1time - name2time, 2)
                name1place = int(race_data.place[race_data.athlete_name == name1])
                name2place = int(race_data.place[race_data.athlete_name == name2])
                if name1place < name2place:
                    winners.append(name1)
                    winner_places.append(name1place)
                    loser_places.append(name2place)
                elif name1place > name2place:
                    winners.append(name2)
                    winner_places.append(name2place)
                    loser_places.append(name1place)
                else:
                    winners.append("Tie")
                    winner_places.append(name1place)
                    loser_places.append(name2place)
                if not math.isnan(diff):
                    table_diffs.append(abs(diff))
                    chart_diffs.append(diff)
                else:
                    table_diffs.append('N/A')

    diff_dict = {
        'winner': winners,
        'winner_place': winner_places,
        'loser_place': loser_places,
        'time_diff': table_diffs,
        'race': races
    }

    df = pd.DataFrame(diff_dict)
    data = df.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in df.columns]
    score = f"{str(winners.count(name1))} - {str(winners.count(name2))}"
    if winners.count("Tie") > 0:
        score = score + f" - {str(winners.count('Tie'))}"

    # create the data and layout for output to figure parameter in graph:

    # data = go.Scatter(
    #     x=chart_diffs,
    #     y=[1 for i in chart_diffs],
    #     mode='markers'
    # )
    #
    # layout = go.Layout(
    #     xaxis={'title': 'Finish Time Difference (s)'}
    # )
    #
    # fig = {
    #     'data': data,
    #     'layout': layout
    # }


    return [dash_table.DataTable(data=data, columns=columns)], score


if __name__ == '__main__':
    app.run_server()
