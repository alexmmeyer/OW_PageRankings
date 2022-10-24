from dash import html, dcc
from dash.dependencies import Input, Output

# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from apps import profile, head2head, progressions, rankings


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Link('PROFILE |', href='/apps/profile'),
        dcc.Link(' PROGRESSIONS |', href='/apps/progressions'),
        dcc.Link(' HEAD-TO-HEAD |', href='/apps/head2head'),
        dcc.Link(' RANKINGS', href='/apps/rankings')
    ], className="row"),
    html.Div(id='page-content', children=[])
])

layouts = {
    '/apps/head2head': head2head.layout,
    '/apps/progressions': progressions.layout,
    '/apps/profile': profile.layout,
    '/apps/rankings': rankings.layout
}


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):

    return layouts[pathname]


if __name__ == '__main__':
    app.run_server(debug=False)