import pandas as pd
import os
import math
from datetime import datetime as dt
from datetime import timedelta, date
import plotly.graph_objs as go
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.express as px

def custom_label(race_result_file, *args):
    race_data = pd.read_csv(race_result_file)
    race_label = ""
    for arg in args:
        race_label = race_label + str(race_data[arg][0]) + " "
    return race_label.strip()

def age_size(race_date, oldest_date):
    min_size = 5
    max_size = 20
    today = date.today()
    period = (today - oldest_date.date()).days
    days_old = (race_date.date() - oldest_date.date()).days
    depreciation = days_old / period
    marker_size = max_size - depreciation*(max_size - min_size)
    return marker_size

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


# Update names in both dropdown lists when a different gender is selected:
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
    diffs = []
    races = []
    dates = []
    distances = []


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
                    loser_places.append('N/A')
                if not math.isnan(diff):
                    diffs.append(diff)
                else:
                    diffs.append('N/A')

    diff_dict = {
        'winner': winners,
        'winner_place': winner_places,
        'loser_place': loser_places,
        'time_diff': diffs,
        'race': races,
        'date': dates,
        'distance (km)': distances,
    }

    oldest_date = min([dt.strptime(i, "%m/%d/%Y") for i in dates])
    print(f"oldest date is {oldest_date}")


    table_df = pd.DataFrame(diff_dict)
    table_df['time_diff'] = [str(timedelta(seconds=abs(i))) if i != 'N/A' else 'N/A' for i in table_df['time_diff']]
    print(table_df)
    data = table_df.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in table_df.columns]
    score = f"{str(winners.count(name1))} - {str(winners.count(name2))}"
    if winners.count("Tie") > 0:
        score = score + f" - {str(winners.count('Tie'))}"

    # create the data and layout for output to figure parameter in graph:

    fig_df = pd.DataFrame(diff_dict)
    fig_df = fig_df[fig_df.time_diff != 'N/A'].reset_index(drop=True)
    fig_df['dt_date'] = [dt.strptime(i, "%m/%d/%Y") for i in fig_df['date']]
    fig_df['marker_size'] = [age_size(fig_df['dt_date'][i], oldest_date) for i in range(len(fig_df))]
    print(fig_df)

    rangemax = max([abs(i) for i in fig_df['time_diff'] if i != 'N/A']) * 1.1

    fig = px.strip(fig_df, x='time_diff', color='winner', hover_data=['race', 'date'])

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

    fig.update_layout(layout)

    return [dash_table.DataTable(data=data, columns=columns)], score, fig


if __name__ == '__main__':
    app.run_server()