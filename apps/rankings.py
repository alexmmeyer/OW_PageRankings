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


# Style dictionaries for dashboard elements:
input_dates_style = {'fontFamily': 'helvetica', 'fontSize': 12, 'display': 'block'}
title_style = {'width': '35%', 'float': 'left', 'display': 'block', 'fontFamily': 'helvetica', 'textAlign': 'center'}
table_div_style = {'width': '35%', 'float': 'center', 'display': 'block'}
date_display_format = 'Y-M-D'

table_formatting = (
            [
                {'if': {
                        'filter_query': '{change} contains "▲"',
                        'column_id': 'change'
                    },
                    'color': 'green'
                },
                {'if': {
                        'filter_query': '{change} contains "▼"',
                        'column_id': 'change'
                    },
                    'color': 'red'
                }
            ]
)

default_ranking_date = '09/30/2022'
default_comparison_date = ''
default_gender = 'women'

# if you change these, also
up_arrow = '▲'
down_arrow = '▼'


layout = html.Div([
    dcc.RadioItems(id='rankings-gender-picker', value=default_gender,
                   options=[{'label': 'Men', 'value': 'men'}, {'label': 'Women', 'value': 'women'}],
                   persistence=True, persistence_type='session'),
    html.Div([
            html.Label('Ranking Date'),
            dcc.DatePickerSingle(id='ranking-date', date=date.today(), display_format=date_display_format,
                                 clearable=False, persistence=True, persistence_type='session'),
            html.Label('Comparison Date'),
            dcc.DatePickerSingle(id='comparison-date', display_format=date_display_format, clearable=True,
                                 persistence=True, persistence_type='session'),
            ], style=input_dates_style),
    html.Div([
            html.H2(id='title-main'),
            html.H3(id='title-date'),
            html.P(id='title-change')
            ], style=title_style),
    html.Div(id='rankings-table', style=table_div_style)
])


@app.callback(
    Output('rankings-table', 'children'),
    [Input('rankings-gender-picker', 'value'),
     Input('ranking-date', 'date'),
     Input('comparison-date', 'date')])
def update_ranking(gender_choice, rank_date, comp_date):
    athlete_countries = pd.read_csv('app_data/' + gender_choice + "/athlete_countries.csv")
    if comp_date is None:
        rankings_directory = 'app_data/' + gender_choice + "/rankings_archive"
        rank_file = f"{rankings_directory}/{rank_date.replace('-', '_')}_{gender_choice}_10km.csv"
        rank_df = pd.read_csv(rank_file)
        rank_df["country"] = [athlete_countries['country'][athlete_countries['athlete_name'] == athlete_name]
                              for athlete_name in rank_df['name']]
        table_df = rank_df[['rank', 'name', 'country']]
        data = table_df.to_dict('rows')
        columns = [{"name": i.title(), "id": i, } for i in table_df.columns]
        table = [dash_table.DataTable(data=data, columns=columns, page_size=100)]
    elif dt.strptime(comp_date, "%Y-%m-%d") >= dt.strptime(rank_date, "%Y-%m-%d"):
        table = "Please choose a comparison date that is prior to the ranking date chosen!"
    else:
        rankings_directory = 'app_data/' + gender_choice + "/rankings_archive"
        rank_file = f"{rankings_directory}/{rank_date.replace('-', '_')}_{gender_choice}_10km.csv"
        rank_df = pd.read_csv(rank_file)
        rank_df["country"] = [athlete_countries['country'][athlete_countries['athlete_name'] == athlete_name]
                              for athlete_name in rank_df['name']]
        comp_file = f"{rankings_directory}/{comp_date.replace('-', '_')}_{gender_choice}_10km.csv"
        comp_df = pd.read_csv(comp_file)

        changes = []

        for athlete_name in rank_df['name']:
            if athlete_name in list(comp_df['name']):
                prev_rank = int(comp_df['rank'][comp_df['name'] == athlete_name])
                rank = int(rank_df['rank'][rank_df['name'] == athlete_name])
                rank_change = prev_rank - rank
                if rank_change == 0:
                    rank_change = '-'
                elif rank_change > 0:
                    rank_change = f"{up_arrow}{rank_change}"
                else:
                    rank_change = f"{down_arrow}{abs(rank_change)}"
            else:
                rank_change = '-'
            changes.append(rank_change)

        rank_df['change'] = changes
        table_df = rank_df[['rank', 'name', 'country', 'change']]
        data = table_df.to_dict('rows')
        columns = [{"name": i.title(), "id": i, } for i in table_df.columns]
        table = [dash_table.DataTable(data=data, columns=columns, page_size=100, style_data_conditional=table_formatting)]

    return table


@app.callback(
    [Output('title-main', 'children'),
     Output('title-date', 'children'),
     Output('title-change', 'children')],
    [Input('rankings-gender-picker', 'value'),
     Input('ranking-date', 'date'),
     Input('comparison-date', 'date')])
def update_title(gender_choice, rank_date, comp_date):

    if comp_date is None:
        title_main = f"{gender_choice}'s 10km Marathon Swimming World Rankings"
        title_main = title_main.upper()
        title_date = dt.strftime(dt.strptime(rank_date, "%Y-%m-%d"), "%d %B, %Y")
        title_change = ''
    elif dt.strptime(comp_date, "%Y-%m-%d") > dt.strptime(rank_date, "%Y-%m-%d"):
        title_main = ''
        title_date = ''
        title_change = ''
    else:
        title_main = f"{gender_choice}'s 10km Marathon Swimming World Rankings"
        title_main = title_main.upper()
        title_date = dt.strftime(dt.strptime(rank_date, "%Y-%m-%d"), "%d %B, %Y")
        formatted_comp_date = dt.strftime(dt.strptime(comp_date, "%Y-%m-%d"), "%d %B, %Y")
        title_change = f"(change since {formatted_comp_date})"

    return title_main, title_date, title_change