"""Export callbacks: PNG, SVG, HTML, CSV, layers JSON."""

from typing import Dict, List, Any

import os
import json
import base64

from dash import dcc, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

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

    # ── Save layers modal: open / cancel ───────────────────────────

    @app.callback(
        Output("save-layers-modal", "is_open"),
        Input("save-layers-btn", "n_clicks"),
        Input("save-layers-cancel", "n_clicks"),
        Input("save-layers-confirm", "n_clicks"),
        State("save-layers-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_save_modal(open_n, cancel_n, confirm_n, is_open):
        triggered = ctx.triggered_id
        if triggered == "save-layers-btn":
            return True
        return False

    # ── Save layers: download JSON on confirm ───────────────────────

    @app.callback(
        Output("download", "data", allow_duplicate=True),
        Input("save-layers-confirm", "n_clicks"),
        State("layers-store", "data"),
        State("active-layer-store", "data"),
        State("save-layers-filename", "value"),
        prevent_initial_call=True,
    )
    def save_layers_file(n_clicks: Any, layers: List, active: Any, filename: str) -> Any:
        """Download all layer configs (without traces) as a JSON file."""
        if not layers:
            raise PreventUpdate
        slim = [
            {k: v for k, v in layer.items() if k != "traces"}
            for layer in layers
        ]
        fname = (filename.strip() or "layers") + ".json"
        payload = json.dumps({"layers": slim, "active": active}, indent=4)
        return dict(content=payload, filename=fname, type="application/json")

    # ── Load layers from uploaded JSON ─────────────────────────────

    @app.callback(
        output={
            "layers": Output("layers-store", "data", allow_duplicate=True),
            "active": Output("active-layer-store", "data", allow_duplicate=True),
        },
        inputs={"contents": Input("upload-layers", "contents")},
        state={"filename": State("upload-layers", "filename")},
        prevent_initial_call=True,
    )
    def load_layers_file(contents: str, filename: str) -> Dict[str, Any]:
        """Parse an uploaded layers JSON and restore the layer store."""
        if contents is None:
            raise PreventUpdate
        _header, encoded = contents.split(",", 1)
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
            data = json.loads(decoded)
        except Exception:
            raise PreventUpdate

        loaded_layers = data.get("layers")
        if not loaded_layers:
            raise PreventUpdate

        # Ensure each layer has an empty traces list
        for layer in loaded_layers:
            layer.setdefault("traces", [])

        active = data.get("active", loaded_layers[0]["id"])
        return {"layers": loaded_layers, "active": active}
