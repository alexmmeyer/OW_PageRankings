#######
# Here we'll use the mpg.csv dataset to demonstrate
# how multiple inputs can affect the same graph.
######
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import csv

df = pd.read_csv('musicians.csv')
print(df)
newrow = ['Pink Floyd', 'David Gilmour', 'Guitar']

with open('musicians.csv', 'a') as file:
    writer = csv.writer(file)
    writer.writerow(newrow)

df = pd.read_csv('musicians.csv')
print(df)