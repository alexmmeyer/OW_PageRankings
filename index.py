import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output

# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from apps import profile, head2head, progressions, rankings, race_trends

link_style = {'display': 'inline'}

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Link('RANKINGS', href='/apps/rankings'),
        dcc.Link('PROFILES', href='/apps/profile'),
        dcc.Link('RANKING PROGRESSIONS', href='/apps/progressions'),
        dcc.Link('RACE TRENDS', href='/apps/race_trends'),
        dcc.Link('HEAD-TO-HEAD', href='/apps/head2head')
    ], className="row", style=link_style),
    html.Div(id='page-content', children=[])
])

# app.layout = html.Div([
#     dcc.Location(id='url', refresh=False),
#     html.Div(
#         dbc.Row(
#             [dbc.Col(dcc.Link('RANKINGS', href='/apps/rankings'), width={'size': 1, 'offset': 3}),
#             dbc.Col(dcc.Link('PROFILES', href='/apps/profile'), width={'size': 1}),
#             dbc.Col(dcc.Link('RANKING PROGRESSIONS', href='/apps/progressions'), width={'size': 2}),
#             dbc.Col(dcc.Link('RACE TRENDS', href='/apps/race_trends'), width={'size': 1}),
#             dbc.Col(dcc.Link('HEAD-TO-HEAD', href='/apps/head2head'), width={'size': 1})]
#         )
#     ),
#     html.Div(id='page-content', children=[])
# ])

layouts = {
    '/apps/head2head': head2head.layout,
    '/apps/progressions': progressions.layout,
    '/apps/profile': profile.layout,
    '/apps/rankings': rankings.layout,
    '/apps/race_trends': race_trends.layout,
    '/': 'Choose a module above.',
}


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):

    return layouts[pathname]


if __name__ == '__main__':
    app.run_server(debug=False)