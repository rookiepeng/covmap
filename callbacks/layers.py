"""Layer management callbacks: add, duplicate, delete, select, render tabs."""

from typing import Dict, List, Any, Optional
import uuid
import copy

import dash
from dash import html, dcc, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

from callbacks.colors import color_for_idx


# Default settings for a new layer
DEFAULT_SETTINGS = {
    "pd": 0.5,
    "pfa": 0.0001,
    "integration": "Swerling 3",
    "rcs": 10,
    "plot": "Azimuth Coverage",
    "inset_position": "top-left",
    "flip": [],
    "fascia": 0,
    "mfg": 0,
    "temp": 0,
    "rain": 0,
    "az_offset": 0,
    "misalign": 0,
    "roll_offset": 0,
    "long_offset": 0,
    "lat_offset": 0,
    "height_offset": 0,
}


def _new_layer_id():
    return str(uuid.uuid4())[:8]


def _collect_settings_from_controls(
    sensor, pd, pfa, integration, rcs, plot, inset_position, flip,
    fascia, mfg, temp, rain, az_offset, misalign, roll_offset,
    long_offset, lat_offset, height_offset, legend,
):
    return {
        "sensor": sensor,
        "pd": pd,
        "pfa": pfa,
        "integration": integration,
        "rcs": rcs,
        "plot": plot,
        "inset_position": inset_position,
        "flip": flip or [],
        "fascia": fascia,
        "mfg": mfg,
        "temp": temp,
        "rain": rain,
        "az_offset": az_offset,
        "misalign": misalign,
        "roll_offset": roll_offset,
        "long_offset": long_offset,
        "lat_offset": lat_offset,
        "height_offset": height_offset,
        "legend": legend or "",
    }


def _build_layer_tabs(layers):
    """Build dbc.Tab items for the layer tab bar."""
    tabs = []
    for idx, layer in enumerate(layers):
        lid = layer["id"]
        name = layer.get("name", "Layer")
        label = f"{idx + 1}. {name}"
        plot_color = color_for_idx(idx)
        tabs.append(
            dbc.Tab(
                label=label,
                tab_id=lid,
                label_style={"fontSize": "0.75rem", "color": "#6c757d"},
                active_label_style={
                    "fontSize": "0.75rem",
                    "color": plot_color,
                    "fontWeight": "600",
                },
            )
        )
    return tabs


def register(app):
    # ── Add / Duplicate / Delete layer ──────────────────────────────

    @app.callback(
        output={
            "layers": Output("layers-store", "data", allow_duplicate=True),
            "active": Output("active-layer-store", "data", allow_duplicate=True),
        },
        inputs={
            "add_btn": Input("add-layer", "n_clicks"),
            "dup_btn": Input("dup-layer", "n_clicks"),
            "del_btn": Input("del-layer", "n_clicks"),
        },
        state={
            "layers": State("layers-store", "data"),
            "active": State("active-layer-store", "data"),
            "sensor": State("sensor", "value"),
            "pd": State("pd", "value"),
            "pfa": State("pfa", "value"),
            "integration": State("integration", "value"),
            "rcs": State("rcs", "value"),
            "plot": State("plot", "value"),
            "inset_position": State("inset-position", "value"),
            "flip": State("flip-checklist", "value"),
            "fascia": State("fascia", "value"),
            "mfg": State("mfg", "value"),
            "temp": State("temp", "value"),
            "rain": State("rain", "value"),
            "az_offset": State("az-offset", "value"),
            "misalign": State("misalign", "value"),
            "roll_offset": State("roll-offset", "value"),
            "long_offset": State("long", "value"),
            "lat_offset": State("lat", "value"),
            "height_offset": State("height", "value"),
            "legend": State("legend", "value"),
        },
        prevent_initial_call=True,
    )
    def manage_layers(
        add_btn, dup_btn, del_btn,
        layers, active,
        sensor, pd, pfa, integration, rcs, plot, inset_position, flip,
        fascia, mfg, temp, rain, az_offset, misalign, roll_offset,
        long_offset, lat_offset, height_offset, legend,
    ):
        triggered = ctx.triggered_id
        layers = layers or []

        if triggered == "add-layer":
            new_id = _new_layer_id()
            new_layer = {
                "id": new_id,
                "name": f"Layer {len(layers) + 1}",
                "settings": dict(DEFAULT_SETTINGS),
                "traces": [],
            }
            layers.append(new_layer)
            return {"layers": layers, "active": new_id}

        elif triggered == "dup-layer":
            if not active or not layers:
                raise PreventUpdate
            src = next((l for l in layers if l["id"] == active), None)
            if src is None:
                raise PreventUpdate
            new_id = _new_layer_id()
            dup = copy.deepcopy(src)
            dup["id"] = new_id
            dup["name"] = src["name"] + " (copy)"
            layers.append(dup)
            return {"layers": layers, "active": new_id}

        elif triggered == "del-layer":
            if not active or not layers:
                raise PreventUpdate
            layers = [l for l in layers if l["id"] != active]
            new_active = layers[-1]["id"] if layers else None
            return {"layers": layers, "active": new_active}

        raise PreventUpdate

    # ── Move active layer left / right ──────────────────────────────

    @app.callback(
        output={
            "layers": Output("layers-store", "data", allow_duplicate=True),
            "replot": Output("replot-trigger", "data", allow_duplicate=True),
        },
        inputs={
            "left_btn": Input("move-layer-left", "n_clicks"),
            "right_btn": Input("move-layer-right", "n_clicks"),
        },
        state={
            "layers": State("layers-store", "data"),
            "active": State("active-layer-store", "data"),
            "replot": State("replot-trigger", "data"),
        },
        prevent_initial_call=True,
    )
    def move_layer(left_btn, right_btn, layers, active, replot):
        triggered = ctx.triggered_id
        layers = layers or []
        if not active or not layers:
            raise PreventUpdate
        idx = next((i for i, l in enumerate(layers) if l["id"] == active), None)
        if idx is None:
            raise PreventUpdate
        if triggered == "move-layer-left":
            if idx == 0:
                raise PreventUpdate
            layers[idx - 1], layers[idx] = layers[idx], layers[idx - 1]
        elif triggered == "move-layer-right":
            if idx == len(layers) - 1:
                raise PreventUpdate
            layers[idx + 1], layers[idx] = layers[idx], layers[idx + 1]
        else:
            raise PreventUpdate
        # Clear cached traces so coverage_plot recolors all layers by new index
        for layer in layers:
            layer["traces"] = []
        return {"layers": layers, "replot": (replot or 0) + 1}

    # ── Select layer via tab click ──────────────────────────────────

    @app.callback(
        Output("active-layer-store", "data", allow_duplicate=True),
        Input("layer-tabs", "active_tab"),
        State("active-layer-store", "data"),
        prevent_initial_call=True,
    )
    def select_layer(active_tab, current_active):
        if not active_tab or active_tab == current_active:
            raise PreventUpdate
        return active_tab

    # ── Populate controls when active layer changes ─────────────────

    @app.callback(
        output={
            "sensor": Output("sensor", "value", allow_duplicate=True),
            "pd": Output("pd", "value", allow_duplicate=True),
            "pfa": Output("pfa", "value", allow_duplicate=True),
            "integration": Output("integration", "value", allow_duplicate=True),
            "rcs": Output("rcs", "value", allow_duplicate=True),
            "plot": Output("plot", "value", allow_duplicate=True),
            "inset_position": Output("inset-position", "value", allow_duplicate=True),
            "flip": Output("flip-checklist", "value", allow_duplicate=True),
            "fascia": Output("fascia", "value", allow_duplicate=True),
            "mfg": Output("mfg", "value", allow_duplicate=True),
            "temp": Output("temp", "value", allow_duplicate=True),
            "rain": Output("rain", "value", allow_duplicate=True),
            "az_offset": Output("az-offset", "value", allow_duplicate=True),
            "misalign": Output("misalign", "value", allow_duplicate=True),
            "roll_offset": Output("roll-offset", "value", allow_duplicate=True),
            "long_offset": Output("long", "value", allow_duplicate=True),
            "lat_offset": Output("lat", "value", allow_duplicate=True),
            "height_offset": Output("height", "value", allow_duplicate=True),
            "legend": Output("legend", "value", allow_duplicate=True),
        },
        inputs={"active": Input("active-layer-store", "data")},
        state={"layers": State("layers-store", "data")},
        prevent_initial_call=True,
    )
    def populate_controls(active, layers):
        if not active or not layers:
            raise PreventUpdate
        layer = next((l for l in layers if l["id"] == active), None)
        if layer is None:
            raise PreventUpdate
        s = layer["settings"]
        saved_sensor = s.get("sensor")
        return {
            "sensor": saved_sensor if saved_sensor is not None else dash.no_update,
            "pd": s.get("pd", 0.5),
            "pfa": s.get("pfa", 0.0001),
            "integration": s.get("integration", "Swerling 3"),
            "rcs": s.get("rcs", 10),
            "plot": s.get("plot", "Azimuth Coverage"),
            "inset_position": s.get("inset_position", "top-left"),
            "flip": s.get("flip", []),
            "fascia": s.get("fascia", 0),
            "mfg": s.get("mfg", 0),
            "temp": s.get("temp", 0),
            "rain": s.get("rain", 0),
            "az_offset": s.get("az_offset", 0),
            "misalign": s.get("misalign", 0),
            "roll_offset": s.get("roll_offset", 0),
            "long_offset": s.get("long_offset", 0),
            "lat_offset": s.get("lat_offset", 0),
            "height_offset": s.get("height_offset", 0),
            "legend": s.get("legend", ""),
        }

    # ── Render layer tabs ───────────────────────────────────────────

    @app.callback(
        Output("layer-tabs", "children"),
        Output("layer-tabs", "active_tab"),
        Input("layers-store", "data"),
        Input("active-layer-store", "data"),
    )
    def render_tabs(layers, active):
        layers = layers or []
        tabs = _build_layer_tabs(layers)
        # Ensure active_tab points to a valid layer
        ids = [l["id"] for l in layers]
        if active not in ids:
            active = ids[-1] if ids else None
        return tabs, active
