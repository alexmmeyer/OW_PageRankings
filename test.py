#######
# Here we'll use the mpg.csv dataset to demonstrate
# how multiple inputs can affect the same graph.
######
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import csv


df = pd.read_csv('women/beatles.csv')
instrument = list(df['instrument'][df['name'] == 'Paul'])[0]
print(str(instrument))
