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

name = 'alex'
names = ['john', 'paul', 'george', 'ringo']
print(names)
print(list(names))
