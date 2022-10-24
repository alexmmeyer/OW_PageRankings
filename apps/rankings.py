import pandas as pd
import os
from datetime import datetime as dt
from datetime import timedelta, date
import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
from app import app

# MM/DD/YYYY string format of today's date
# today = dt.strftime(date.today(), "%m/%d/%Y")
today = '09/30/2022'

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
title_style = {'width': '35%', 'float': 'left', 'display': 'block', 'fontFamily': 'helvetica', 'textAlign': 'center'}
table_div_style = {'width': '35%', 'float': 'center', 'display': 'block'}

default_ranking_date = '09/30/2022'
default_comparison_date = '08/31/2022'
default_gender = 'women'


layout = html.Div([
    dcc.RadioItems(id='rankings-gender-picker', value=default_gender,
                   options=[{'label': 'Men', 'value': 'men'}, {'label': 'Women', 'value': 'women'}],
                   persistence=True, persistence_type='session'),
    html.Div([
            html.Label('Ranking Date (MM/DD/YYYY)'),
            dcc.Input(id='ranking-date', value=default_ranking_date, persistence=True, persistence_type='session'),
            html.Label('Comparison Date (MM/DD/YYYY)'),
            dcc.Input(id='comparison-date', value=default_comparison_date, persistence=True, persistence_type='session'),
            ], style=input_dates_style),
    html.Div([
            html.H2(id='title-main'),
            html.H3(id='title-date'),
            html.P(id='title-change')
            ], style=title_style),
    html.Div(id='rankings-table', children='hello world!', style=table_div_style)
])


@app.callback(
    Output('rankings-table', 'children'),
    [Input('rankings-gender-picker', 'value'),
     Input('ranking-date', 'value'),
     Input('comparison-date', 'value')])
def update_ranking(gender_choice, rank_date, comp_date):
    if dt.strptime(comp_date, "%m/%d/%Y") > dt.strptime(rank_date, "%m/%d/%Y"):
        table = "Please choose a comparison date that is prior to the ranking date chosen!"
    else:
        rankings_directory = 'app_data/' + gender_choice + "/rankings_archive"
        rank_file = f"{rankings_directory}/{alpha_date(rank_date)}_{gender_choice}_10km.csv"
        rank_df = pd.read_csv(rank_file)
        comp_file = f"{rankings_directory}/{alpha_date(comp_date)}_{gender_choice}_10km.csv"
        comp_df = pd.read_csv(comp_file)

        changes = []

        for athlete_name in rank_df['name']:
            if athlete_name in list(comp_df['name']):
                prev_rank = int(comp_df['rank'][comp_df['name'] == athlete_name])
                rank = int(rank_df['rank'][rank_df['name'] == athlete_name])
                rank_change = prev_rank - rank
                if rank_change == 0:
                    rank_change = '-'
            else:
                rank_change = '-'
            changes.append(rank_change)

        table_df = rank_df
        table_df['change'] = changes
        table_df = table_df.drop('pagerank', axis='columns')
        data = table_df.to_dict('rows')
        columns = [{"name": i, "id": i, } for i in table_df.columns]
        table = [dash_table.DataTable(data=data, columns=columns)]

    return table


@app.callback(
    [Output('title-main', 'children'),
     Output('title-date', 'children'),
     Output('title-change', 'children')],
    [Input('rankings-gender-picker', 'value'),
     Input('ranking-date', 'value'),
     Input('comparison-date', 'value')])
def update_title(gender_choice, rank_date, comp_date):

    title_main = f"{gender_choice}'s 10km Marathon Swimming World Rankings"
    title_main = title_main.upper()
    title_date = rank_date
    title_change = f"(change since {comp_date})"

    return title_main, title_date, title_change
