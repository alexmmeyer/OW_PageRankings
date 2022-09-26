import pandas as pd
import os
import math
from datetime import datetime as dt
from datetime import timedelta, date
import plotly.graph_objs as go
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output

def custom_label(race_result_file, *args):
    race_data = pd.read_csv(race_result_file)
    race_label = ""
    for arg in args:
        race_label = race_label + str(race_data[arg][0]) + " "
    return race_label.strip()

def age_opacity(race_date, oldest_date):
    min_opacity = .25
    max_opacity = .75
    today = date.today()
    period = (today - oldest_date.date()).days
    days_old = (race_date.date() - oldest_date.date()).days
    depreciation = days_old / period
    opacity = max_opacity - depreciation*(max_opacity - min_opacity)
    return opacity

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
    dates = []
    distances = []
    dt_diffs = []

    chart_diffs = []

    for file in os.listdir(results_directory):
        results_file_path = os.path.join(results_directory, file)
        race_data = pd.read_csv(results_file_path)
        race_dist = race_data.distance[0]
        if name1 in list(race_data.athlete_name) and name2 in list(race_data.athlete_name):
            if race_dist == dist or dist == "all":
                race_name = custom_label(os.path.join(results_directory, file), "event", "location")
                date = race_data["date"][0]
                distance = race_data['distance'][0]
                races.append(race_name)
                dates.append(date)
                distances.append(distance)
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
                    table_diffs.append(str(timedelta(seconds=abs(diff))))
                    chart_diffs.append(diff)
                    dt_diffs.append(str(timedelta(seconds=diff)))
                else:
                    table_diffs.append('N/A')
                    chart_diffs.append('N/A')
                    dt_diffs.append('N/A')

    diff_dict = {
        'winner': winners,
        'winner_place': winner_places,
        'loser_place': loser_places,
        'time_diff': table_diffs,
        'race': races,
        'date': dates,
        'distance (km)': distances,
        'dt_diff': dt_diffs
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
    df2['dt_date'] = [dt.strptime(i, "%m/%d/%Y") for i in df2['date']]
    # df2['time_diff'] = [timedelta(seconds=i) for i in df2['time_diff']]
    print(df2)

    chart_data = []
    colors = {
        name1: 'blue',
        name2: 'green',
        "Tie": 'orange'
    }
    unique_winners = list(df2['winner'].unique())
    oldest_date = min(df2['dt_date'])
    opacities = [age_opacity(i, oldest_date) for i in df2['dt_date']]
    # think i should try plotly.express for this because for loop below is based on traces (unique winners / tie), not rows of df2

    for i in unique_winners:
        df3 = df2[df2['winner'] == i]
        trace = go.Scatter(
                    x=df3['time_diff'],
                    y=['str' for i in df3['time_diff']],
                    mode='markers',
                    marker={'size': 20, 'line': {'width': 0.5, 'color': 'black'}, 'opacity': 0.5, 'color': colors[i]},
                    name=i
    )
        chart_data.append(trace)

    rangemax = max([abs(i) for i in chart_diffs if i != 'N/A']) * 1.1

    layout = go.Layout(
        xaxis={'title': 'Finish Time Difference (s)', 'range': [-rangemax, rangemax]},
        yaxis={'showticklabels': False},
        legend_title_text='Winner:',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=.5
        )
    )


    fig = {
        'data': chart_data,
        'layout': layout
    }

    return [dash_table.DataTable(data=data, columns=columns)], score, fig


if __name__ == '__main__':
    app.run_server()