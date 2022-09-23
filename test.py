#######
# Here we'll use the mpg.csv dataset to demonstrate
# how multiple inputs can affect the same graph.
######
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

app = Dash()

df = pd.read_csv('../Plotly-Dashboards-with-Dash-master/Data/mpg.csv')

features = df.columns

diffs = [-1, 2, 6, -5, 7, -8]
print(max([abs(diff) for diff in diffs]))
