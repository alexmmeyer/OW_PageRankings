import pandas as pd
import os
from datetime import datetime as dt
from datetime import timedelta
from datetime import date
from itertools import combinations
import variables
import networkx as nx
import time
import matplotlib.pyplot as plt
import math
import seaborn as sb
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def alpha_date(date):
    """
    :param date: MM/DD/YYYY
    :return: YYYY_MM_DD
    """
    date = date.replace("/", "_")
    alphadate = date[6:] + "_" + date[:5]
    return alphadate


def get_results(athlete_name):
    rows = []

    for file in os.listdir(RESULTS_DIRECTORY):
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        race_data = pd.read_csv(results_file_path)
        names_list = list(race_data.athlete_name)
        names_list = [name.title() for name in names_list]
        race_data.athlete_name = names_list
        if athlete_name.title() in names_list:
            row = race_data[race_data.athlete_name == athlete_name.title()]
            rows.append(row)
            # print(file)
            # print(row)

    return pd.concat(rows, ignore_index=True)

RESULTS_DIRECTORY = variables.RESULTS_DIRECTORY
RANKINGS_DIRECTORY = variables.RANKINGS_DIRECTORY
RANKING_FILE_NAME = variables.RANKING_FILE_NAME
LAMBDA = variables.LAMBDA
DEPRECIATION_MODEL = variables.DEPRECIATION_MODEL
GENDER = variables.GENDER
if GENDER == "men":
    DEPRECIATION_PERIOD = variables.M_DEPRECIATION_PERIOD
    K = variables.M_K
elif GENDER == "women":
    DEPRECIATION_PERIOD = variables.W_DEPRECIATION_PERIOD
    K = variables.W_K
RANK_DIST = variables.RANK_DIST
FROM_RANK = variables.FROM_RANK
TO_RANK = variables.TO_RANK
event_type_weights = variables.event_weights
athlete_countries = variables.athlete_countries

def ranking_progression_multi(start_date, end_date, athlete_names):
    """
    :param athlete_name:
    :param start_date:
    :param end_date:
    :return: graph showing athlete's ranking on every day between (inclusive) start_date and end_date
    :param increment:
    """

    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    # Get a list of dates called date_range within the start and end range
    increment = 1
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    traces = []

    for athlete_name in athlete_names:
        # Loop through each of the dates in date_range and look up the ranking for that date in the archive. Add the date
        # to one list and add the athlete's rank to a separate list. Count loops to track progress.
        athlete_name = athlete_name.title()
        rank_dates = []
        ranks = []

        # get the data for the progression line:
        for date in date_range:
            file_name = f"{alpha_date(date)}_{GENDER}_{RANK_DIST}km.csv"
            ranking_data = pd.read_csv(f"{RANKINGS_DIRECTORY}/{file_name}")
            ranked_athletes = list(ranking_data.name)
            if athlete_name in ranked_athletes:
                rank_dates.append(dt.strptime(date, "%m/%d/%Y"))
                rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
                ranks.append(rank_on_date)
                # print(f'{athlete_name} was ranked {rank_on_date} on {date}')
        line_trace = go.Scatter(x=rank_dates,
                           y=ranks,
                           mode='lines',
                           opacity=0.8,
                           name=athlete_name)
        traces.append(line_trace)

        results_df = get_results(athlete_name)
        results_df["dt_date"] = [dt.strptime(date, "%m/%d/%Y") for date in results_df.date]
        results_df = results_df[results_df.dt_date >= start_date]
        results_df = results_df[results_df.dt_date <= end_date]
        race_date_ranks = []
        for date in results_df.date:
            file_name = f"{alpha_date(date)}_{GENDER}_{RANK_DIST}km.csv"
            ranking_data = pd.read_csv(f"{RANKINGS_DIRECTORY}/{file_name}")
            rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
            race_date_ranks.append(rank_on_date)
        results_df['rank'] = race_date_ranks
        print(results_df)

        scatter_trace = go.Scatter(x=results_df['dt_date'],
                                   y=results_df['rank'],
                                   mode='markers',
                                   marker={'size': 10},
                                   name=athlete_name)

        traces.append(scatter_trace)

    layout = go.Layout(
            xaxis={'title': 'Date'},
            yaxis={'title': 'World Ranking'},
            hovermode='closest')

    fig = go.Figure(data=traces, layout=layout)
    pyo.plot(fig)

    # return {
    #     'data': traces,
    #     'layout': go.Layout(
    #         xaxis={'type': 'log', 'title': 'GDP Per Capita'},
    #         yaxis={'title': 'Life Expectancy'},
    #         hovermode='closest'
    #     )
    # }

# print('hello')
ranking_progression_multi("01/01/2022", "07/31/2022", ["Lea Boy", "Barbara Pozzobon", "Caroline Laure Jouisse"])
