from dash import html, dcc
from dash.dependencies import Input, Output

# Connect to main app.py file
from app import app
from app import server

# Connect to app pages
from apps import profile, multiprogression, head2head

app.layout = html.Div([
    html.Div([
        dcc.Link('PROFILE |', href='/apps/profile'),
        dcc.Link(' PROGRESSIONS |', href='/apps/multiprogression'),
        dcc.Link(' HEAD-TO-HEAD', href='/apps/head2head')
    ], className="row"),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=[])
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    return pathname
    # if pathname == '/apps/profile':
    #     return profile.layout
    # elif pathname == '/apps/multiprogression':
    #     return multiprogression.layout
    # elif pathname == '/apps/head2head':
    #     return head2head.layout
    # else:
    #     return "404 Page Error! Please choose a link"


if __name__ == '__main__':
    app.run_server(debug=False)
