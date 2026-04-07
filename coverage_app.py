"""
Copyright (C) 2023 - PRESENT  Zhengyu Peng

Coverage Map Application - A tool for visualizing radar coverage patterns.
"""

from flaskwebgui import FlaskUI

import dash

from layout.layout import get_app_layout
from callbacks import register_all

app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width,height=device-height,initial-scale=1",
        }
    ],
)

app.scripts.config.serve_locally = True
app.css.config.serve_locally = True
app.title = "Coverage Map"
app.layout = get_app_layout
server = app.server

register_all(app)


if __name__ == "__main__":
    FlaskUI(
        app=server, server="flask", port=34687, profile_dir_prefix="coverage_app"
    ).run()
