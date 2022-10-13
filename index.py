from dash import html, dcc
from dash.dependencies import Input, Output

# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from apps import profile, head2head, progressions


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Link('PROFILE |', href='/apps/profile'),
        dcc.Link(' PROGRESSIONS |', href='/apps/progressions'),
        dcc.Link(' HEAD-TO-HEAD', href='/apps/head2head')
    ], className="row"),
    html.Div(id='page-content', children=[])
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/head2head':
        return head2head.layout
    if pathname == '/apps/progressions':
        return progressions.layout
    if pathname == '/apps/profile':
        return profile.layout
    else:
        return "Please choose a link"


if __name__ == '__main__':
    app.run_server(debug=False)