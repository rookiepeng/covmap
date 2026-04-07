"""Export callbacks: PNG, SVG, HTML, CSV."""

from typing import Dict, List, Any

import os

from dash import dcc
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import pandas as pds


def register(app):
    @app.callback(
        Output("download", "data", allow_duplicate=True),
        Input("export-png", "n_clicks"),
        State("scatter", "figure"),
        prevent_initial_call=True,
    )
    def export_png(unused_n_clicks: Any, fig: Dict[str, Any]) -> Any:
        """Export the current figure as a PNG file."""
        figure = go.Figure(fig)
        if not os.path.exists("temp"):
            os.mkdir("temp")
        figure.write_image("./temp/plot.png", scale=2)
        return dcc.send_file("./temp/plot.png")

    @app.callback(
        Output("download", "data", allow_duplicate=True),
        Input("export-svg", "n_clicks"),
        State("scatter", "figure"),
        prevent_initial_call=True,
    )
    def export_svg(unused_n_clicks: Any, fig: Dict[str, Any]) -> Any:
        """Export the current figure as an SVG file."""
        figure = go.Figure(fig)
        if not os.path.exists("temp"):
            os.mkdir("temp")
        figure.write_image("./temp/plot.svg")
        return dcc.send_file("./temp/plot.svg")

    @app.callback(
        Output("download", "data", allow_duplicate=True),
        Input("export-html", "n_clicks"),
        State("scatter", "figure"),
        prevent_initial_call=True,
    )
    def export_html(unused_n_clicks: Any, fig: Dict[str, Any]) -> Any:
        """Export the current figure as an HTML file."""
        figure = go.Figure(fig)
        if not os.path.exists("temp"):
            os.mkdir("temp")
        figure.write_html("./temp/plot.html")
        return dcc.send_file("./temp/plot.html")

    @app.callback(
        Output("download", "data", allow_duplicate=True),
        Input("export-data", "n_clicks"),
        [
            State("layers-store", "data"),
            State("active-layer-store", "data"),
            State("plot", "value"),
        ],
        prevent_initial_call=True,
    )
    def export_data(
        unused_n_clicks: Any, layers: List, active: Any, plot_type: str
    ) -> Any:
        """Export the raw data of the active layer as a CSV file."""
        from dash.exceptions import PreventUpdate

        if not layers or not active:
            raise PreventUpdate
        layer = next((l for l in layers if l["id"] == active), None)
        if layer is None or not layer.get("traces"):
            raise PreventUpdate
        data = layer["traces"]
        if plot_type == "Azimuth Coverage":
            dataframe = pds.DataFrame(
                {"longitude_m": data[0]["x"], "latitude_m": data[0]["y"]}
            )
        elif plot_type == "Azimuth vs. Range":
            dataframe = pds.DataFrame(
                {"azimuth_deg": data[0]["x"], "range_m": data[0]["y"]}
            )
        elif plot_type == "Elevation Coverage":
            dataframe = pds.DataFrame(
                {"longitude_m": data[0]["x"], "height_m": data[0]["y"]}
            )
        elif plot_type == "Elevation vs. Range":
            dataframe = pds.DataFrame(
                {"elevation_deg": data[0]["x"], "range_m": data[0]["y"]}
            )

        if not os.path.exists("temp"):
            os.mkdir("temp")

        return dcc.send_data_frame(dataframe.to_csv, "plot_data.csv")
