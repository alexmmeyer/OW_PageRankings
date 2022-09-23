import pandas as pd
import os
import math
from datetime import datetime as dt
from datetime import timedelta
import plotly.graph_objs as go
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output


app = Dash()

score_style = {'fontFamily': 'helvetica', 'fontSize': 96, 'textAlign': 'center'}

dropdown_div_style = {'width': '50%', 'float': 'left', 'display': 'block'}

app.layout = html.Div([
    dcc.RadioItems(id='gender-picker', options=[{'label': 'Men', 'value': 'men'}, {'label': 'Women', 'value': 'women'}], value='women'),
    html.Div([
        html.Div(dcc.Dropdown(id='name-dropdown1', value='Lea Boy',
                     options=[{'label': i, 'value': i} for i in pd.read_csv("women/athlete_countries.csv")]), style=dropdown_div_style),
        html.Div(dcc.Dropdown(id='name-dropdown2', value='Caroline Laure Jouisse',
                     options=[{'label': i, 'value': i} for i in pd.read_csv("women/athlete_countries.csv")]), style=dropdown_div_style)]),
        html.H1(id='score', style=score_style),
        dcc.Graph(id='diff-graph'),
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
               Output('score', 'children'),
               Output('diff-graph', 'figure')],
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
    print(df)
    data = df.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in df.columns]
    score = f"{str(winners.count(name1))} - {str(winners.count(name2))}"
    if winners.count("Tie") > 0:
        score = score + f" - {str(winners.count('Tie'))}"

    # create the data and layout for output to figure parameter in graph:

    df['time_diff'] = chart_diffs
    df2 = df[df.time_diff != 'N/A'].reset_index(drop=True)

    chart_data = []
    colors = ['blue', 'green', 'orange']
    unique_winners = list(df2['winner'].unique())

    for i in unique_winners:
        colorindex = unique_winners.index(i)
        df3 = df2[df2['winner'] == i]
        trace = go.Scatter(
                    x=df3['time_diff'],
                    y=['str' for i in df3['time_diff']],
                    mode='markers',
                    marker={'size': 20, 'line': {'width': 0.5, 'color': 'black'}, 'opacity': 0.5, 'color': colors[colorindex]}
    )
        chart_data.append(trace)

    rangemax = max([abs(i) for i in chart_diffs]) * 1.1

    layout = go.Layout(
        xaxis={'title': 'Finish Time Difference (s)', 'range': [-rangemax, rangemax]}
    )


    fig = {
        'data': chart_data,
        'layout': layout
    }

    return [dash_table.DataTable(data=data, columns=columns)], score, fig


if __name__ == '__main__':
    app.run_server()