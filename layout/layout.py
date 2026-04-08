"""
    Copyright (C) 2023 - PRESENT  Zhengyu Peng
    
    This module defines the layout and UI components for a radar coverage mapping application.
    It creates a responsive web interface using Dash and Bootstrap components.
"""

from dash import dcc
from dash import html

import dash_bootstrap_components as dbc

import json
import os

import plotly.io as pio


# Define available target integration/model types
INTEGRATION = [
    "Swerling 0",  # Constant RCS target
    "Swerling 1",  # Fluctuating RCS, scan-to-scan variation
    "Swerling 2",  # Fluctuating RCS, pulse-to-pulse variation
    "Swerling 3",  # Fluctuating RCS, scan-to-scan variation with higher mean
    "Swerling 4",  # Fluctuating RCS, pulse-to-pulse variation with higher mean
    "Coherent",    # Coherent integration
]

# Export menu configuration - allows users to export plots in different formats
export_menu_items = [
    dbc.DropdownMenuItem("Export PNG", id="export-png"),
    dbc.Tooltip(
        "Export the static figure",
        target="export-png",
        placement="top",
    ),
    dbc.DropdownMenuItem("Export SVG", id="export-svg"),
    dbc.Tooltip(
        "Export the static vector figure",
        target="export-svg",
        placement="top",
    ),
    dbc.DropdownMenuItem("Export HTML", id="export-html"),
    dbc.Tooltip(
        "Export the interactive figure",
        target="export-html",
        placement="top",
    ),
    dbc.DropdownMenuItem(divider=True),
    dbc.DropdownMenuItem("Export Data", id="export-data"),
    dbc.Tooltip(
        "Export data of the latest plot",
        target="export-data",
        placement="top",
    ),
]

# Checklist for pattern manipulation
# Allows users to flip azimuth or elevation patterns for better visualization
flip_checklist = dbc.Checklist(
    options=[
        {"label": "Flip Azimuth", "value": "flip_az"},
        {"label": "Flip Elevation", "value": "flip_el"},
        {"label": "Fill", "value": "fill"},
    ],
    value=["fill"],
    id="flip-checklist",
    inline=True,
)

# Target RCS (Radar Cross Section) control
# Allows adjustment of target RCS from -30 to 30 dBsm
rcs_slider = html.Div(
    [
        html.Div(
            [html.Span("RCS", className="fw-semibold", id="lbl-rcs"), html.Span(" (dBsm)", className="text-muted"),
             dbc.Tooltip("Target's RCS", target="lbl-rcs", placement="top")],
            className="small mb-1",
        ),
        dcc.Slider(
            id="rcs",
            min=-30,
            max=30,
            step=0.1,
            value=10,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "top"},
        ),
    ],
    className="mb-3",
)

# Fascia loss control
# Accounts for signal loss due to vehicle fascia/radome
fascia_slider = html.Div(
    [
        html.Div(
            [html.Span("Fascia", className="fw-semibold", id="lbl-fascia"), html.Span(" (dB)", className="text-muted"),
             dbc.Tooltip("Fascia loss", target="lbl-fascia", placement="top")],
            className="small mb-1",
        ),
        dcc.Slider(
            id="fascia",
            min=-10,
            max=0,
            step=0.1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "top"},
        ),
    ],
    className="mb-3",
)

# Manufacturing margin control
# Accounts for manufacturing variations and tolerances
mfg_slider = html.Div(
    [
        html.Div(
            [html.Span("MFG", className="fw-semibold", id="lbl-mfg"), html.Span(" (dB)", className="text-muted"),
             dbc.Tooltip("Manufacturing margin", target="lbl-mfg", placement="top")],
            className="small mb-1",
        ),
        dcc.Slider(
            id="mfg",
            min=-10,
            max=0,
            step=0.1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "top"},
        ),
    ],
    className="mb-3",
)

# Temperature-related loss control
# Accounts for performance variations due to temperature
temp_slider = html.Div(
    [
        html.Div(
            [html.Span("Temp", className="fw-semibold", id="lbl-temp"), html.Span(" (dB)", className="text-muted"),
             dbc.Tooltip("Temperature loss", target="lbl-temp", placement="top")],
            className="small mb-1",
        ),
        dcc.Slider(
            id="temp",
            min=-10,
            max=0,
            step=0.1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "top"},
        ),
    ],
    className="mb-3",
)

# Rain-induced loss control
# Accounts for signal attenuation due to rain
rain_slider = html.Div(
    [
        html.Div(
            [html.Span("Rain", className="fw-semibold", id="lbl-rain"), html.Span(" (dB)", className="text-muted"),
             dbc.Tooltip("Rain damping loss", target="lbl-rain", placement="top")],
            className="small mb-1",
        ),
        dcc.Slider(
            id="rain",
            min=-10,
            max=0,
            step=0.1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "top"},
        ),
    ],
    className="mb-3",
)

# Elevation misalignment control
# Allows adjustment for mounting angle errors
misalign_slider = html.Div(
    [
        html.Div(
            [html.Span("Pitch offset", className="fw-semibold", id="lbl-misalign"), html.Span(" (deg)", className="text-muted"),
             dbc.Tooltip("Pitch misalignment", target="lbl-misalign", placement="top")],
            className="small mb-1",
        ),
        dcc.Slider(
            id="misalign",
            min=-20,
            max=20,
            step=1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "top"},
        ),
    ],
    className="mb-3",
)

# Roll offset control
# Allows adjustment of roll angle around boresight axis
roll_slider = html.Div(
    [
        html.Div(
            [html.Span("Roll offset", className="fw-semibold", id="lbl-roll"), html.Span(" (deg)", className="text-muted"),
             dbc.Tooltip("Roll offset around boresight", target="lbl-roll", placement="top")],
            className="small mb-1",
        ),
        dcc.Slider(
            id="roll-offset",
            min=-90,
            max=90,
            step=1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "top"},
        ),
    ],
    className="mb-3",
)

# Azimuth offset control
# Allows adjustment of azimuth angle offset
az_slider = html.Div(
    [
        html.Div(
            [html.Span("Yaw offset", className="fw-semibold", id="lbl-az"), html.Span(" (deg)", className="text-muted"),
             dbc.Tooltip("Yaw offset", target="lbl-az", placement="top")],
            className="small mb-1",
        ),
        dcc.Slider(
            id="az-offset",
            min=-180,
            max=180,
            step=1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "top"},
        ),
    ],
    className="mb-3",
)

# Longitudinal position offset control
# Adjusts sensor position along vehicle's length
long_slider = html.Div(
    [
        html.Div(
            [html.Span("Long", className="fw-semibold", id="lbl-long"), html.Span(" (m)", className="text-muted"),
             dbc.Tooltip("Longitudinal offset", target="lbl-long", placement="top")],
            className="small mb-1",
        ),
        dcc.Slider(
            id="long",
            min=-5,
            max=5,
            step=0.001,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "top"},
        ),
    ],
    className="mb-3",
)

# Lateral position offset control
# Adjusts sensor position across vehicle's width
lat_slider = html.Div(
    [
        html.Div(
            [html.Span("Lat", className="fw-semibold", id="lbl-lat"), html.Span(" (m)", className="text-muted"),
             dbc.Tooltip("Latitudinal offset", target="lbl-lat", placement="top")],
            className="small mb-1",
        ),
        dcc.Slider(
            id="lat",
            min=-5,
            max=5,
            step=0.001,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "top"},
        ),
    ],
    className="mb-3",
)

# Height offset control
# Adjusts sensor mounting height
height_slider = html.Div(
    [
        html.Div(
            [html.Span("Height", className="fw-semibold", id="lbl-height"), html.Span(" (m)", className="text-muted"),
             dbc.Tooltip("Height offset", target="lbl-height", placement="top")],
            className="small mb-1",
        ),
        dcc.Slider(
            id="height",
            min=-5,
            max=5,
            step=0.001,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "top"},
        ),
    ],
    className="mb-3",
)

# Control and chart panels as separate cards
plot_card = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                                dbc.InputGroup(
                                    [
                                        dbc.Button(
                                            html.I(className="bi bi-record-circle"),
                                            id="reload",
                                            n_clicks=0,
                                            color="primary",
                                            disabled=False,
                                        ),
                                        dbc.Select(id="sensor"),
                                        dbc.Tooltip(
                                            "Pick a sensor",
                                            target="sensor",
                                            placement="top",
                                        ),
                                        dcc.Upload(
                                            id="upload-config",
                                            children=dbc.Button(
                                                html.I(className="bi bi-upload"),
                                                color="success",
                                                style={
                                                    "border-top-left-radius": "0",
                                                    "border-bottom-left-radius": "0",
                                                },
                                            ),
                                            accept="application/json",
                                        ),
                                        dbc.Tooltip(
                                            "Upload a new sensor configuration",
                                            target="upload-config",
                                            placement="top",
                                        ),
                                    ],
                                    className="mb-2",
                                    size="sm",
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            id="property-container",
                                            children=[],
                                            className="mb-3",
                                        ),
                                        dbc.Accordion(
                                            [
                                                dbc.AccordionItem(
                                                    [
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Pd"),
                                                                dbc.Input(
                                                                    id="pd",
                                                                    type="number",
                                                                    value=0.5,
                                                                    min=0.01,
                                                                    max=0.9999,
                                                                    step=0.0001,
                                                                ),
                                                                dbc.Tooltip(
                                                                    "Probability of detection",
                                                                    target="pd",
                                                                    placement="top",
                                                                ),
                                                            ],
                                                            size="sm",
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Pfa"),
                                                                dbc.Input(
                                                                    id="pfa",
                                                                    type="number",
                                                                    value=0.0001,
                                                                    min=0.00000000001,
                                                                    max=0.1,
                                                                    step=0.00000000001,
                                                                ),
                                                                dbc.Tooltip(
                                                                    "Probability of false alarm",
                                                                    target="pfa",
                                                                    placement="top",
                                                                ),
                                                            ],
                                                            size="sm",
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Target"),
                                                                dbc.Select(
                                                                    id="integration",
                                                                    options=[
                                                                        {"label": i, "value": i}
                                                                        for i in INTEGRATION
                                                                    ],
                                                                    value="Swerling 3",
                                                                ),
                                                                dbc.Tooltip(
                                                                    "Target model",
                                                                    target="integration",
                                                                    placement="top",
                                                                ),
                                                            ],
                                                            size="sm",
                                                            className="mb-2",
                                                        ),
                                                        rcs_slider,
                                                    ],
                                                    title="Detection",
                                                    item_id="detection",
                                                ),
                                                dbc.AccordionItem(
                                                    [
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Plot"),
                                                                dbc.Select(
                                                                    id="plot",
                                                                    options=[
                                                                        "Azimuth Coverage",
                                                                        "Azimuth vs. Range",
                                                                        "Elevation Coverage",
                                                                        "Elevation vs. Range",
                                                                    ],
                                                                    value="Azimuth Coverage",
                                                                ),
                                                                dbc.Tooltip(
                                                                    "Figure configuration",
                                                                    target="plot",
                                                                    placement="top",
                                                                ),
                                                            ],
                                                            size="sm",
                                                            className="mb-2",
                                                        ),
                                                        dbc.InputGroup(
                                                            [
                                                                dbc.InputGroupText("Inset"),
                                                                dbc.Select(
                                                                    id="inset-position",
                                                                    options=[
                                                                        {"label": "Hidden", "value": "hidden"},
                                                                        {"label": "Top-Right", "value": "top-right"},
                                                                        {"label": "Top-Left", "value": "top-left"},
                                                                        {"label": "Bottom-Right", "value": "bottom-right"},
                                                                        {"label": "Bottom-Left", "value": "bottom-left"},
                                                                    ],
                                                                    value="top-left",
                                                                ),
                                                                dbc.Tooltip(
                                                                    "Inset heatmap position",
                                                                    target="inset-position",
                                                                    placement="top",
                                                                ),
                                                            ],
                                                            size="sm",
                                                            className="mb-2",
                                                        ),
                                                        flip_checklist,
                                                    ],
                                                    title="Plot",
                                                    item_id="plot-settings",
                                                ),
                                                dbc.AccordionItem(
                                                    [
                                                        fascia_slider,
                                                        mfg_slider,
                                                        temp_slider,
                                                        rain_slider,
                                                    ],
                                                    title="Losses",
                                                    item_id="losses",
                                                ),
                                                dbc.AccordionItem(
                                                    [
                                                        az_slider,
                                                        misalign_slider,
                                                        roll_slider,
                                                    ],
                                                    title="Orientation",
                                                    item_id="orientation",
                                                ),
                                                dbc.AccordionItem(
                                                    [
                                                        long_slider,
                                                        lat_slider,
                                                        height_slider,
                                                    ],
                                                    title="Position",
                                                    item_id="position",
                                                ),
                                            ],
                                            active_item=[
                                                "detection",
                                                "plot-settings",
                                                "losses",
                                                "orientation",
                                            ],
                                            always_open=True,
                                            flush=False,
                                        ),
                                    ],
                                    style={
                                        "overflowY": "auto",
                                        "overflowX": "hidden",
                                        "flex": "1 1 0",
                                        "minHeight": "0",
                                        "padding": "4px",
                                    },
                                ),
                            ],
                    className="p-2",
                ),
                className="control-card h-100 p-2",
            ),
            width=3,
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                                html.Div(
                                    [
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button(html.I(className="bi bi-plus-lg"), id="add-layer", n_clicks=0, color="success", size="sm"),
                                                dbc.Tooltip("Add a new layer", target="add-layer", placement="top"),
                                                dbc.Button(html.I(className="bi bi-files"), id="dup-layer", n_clicks=0, color="primary", size="sm"),
                                                dbc.Tooltip("Duplicate active layer", target="dup-layer", placement="top"),
                                                dbc.Button(html.I(className="bi bi-trash"), id="del-layer", n_clicks=0, color="danger", size="sm"),
                                                dbc.Tooltip("Delete active layer", target="del-layer", placement="top"),
                                            ],
                                        ),
                                        html.Div(style={"width": "1px", "background": "#dee2e6", "alignSelf": "stretch", "margin": "0 6px"}),
                                        html.Div(
                                            [
                                                dcc.Upload(
                                                    id="upload-layers",
                                                    children=dbc.Button(
                                                        [html.I(className="bi bi-folder2-open"), " Load"],
                                                        color="secondary", size="sm",
                                                        style={"borderRadius": "4px 0 0 4px"},
                                                    ),
                                                    accept="application/json",
                                                    style={"display": "inline-block"},
                                                ),
                                                dbc.Button(
                                                    [html.I(className="bi bi-download"), " Save"],
                                                    id="save-layers-btn", n_clicks=0, color="secondary", size="sm",
                                                    style={"borderRadius": "0 4px 4px 0", "marginLeft": "-1px"},
                                                ),
                                                dbc.Tooltip("Load layers from a JSON file", target="upload-layers", placement="top"),
                                                dbc.Tooltip("Save all layer configs to a JSON file", target="save-layers-btn", placement="top"),
                                            ],
                                            className="d-flex",
                                        ),
                                        html.Div(className="ms-auto"),
                                        dbc.DropdownMenu(export_menu_items, label="Export", color="info", size="sm"),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader(dbc.ModalTitle("Save Layers")),
                                                dbc.ModalBody(
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText(html.I(className="bi bi-file-earmark-code")),
                                                            dbc.Input(id="save-layers-filename", value="layers", placeholder="filename (no extension)"),
                                                            dbc.InputGroupText(".json"),
                                                        ],
                                                        size="sm",
                                                    ),
                                                ),
                                                dbc.ModalFooter(
                                                    [
                                                        dbc.Button("Download", id="save-layers-confirm", color="primary", size="sm"),
                                                        dbc.Button("Cancel", id="save-layers-cancel", color="secondary", size="sm", className="ms-2"),
                                                    ]
                                                ),
                                            ],
                                            id="save-layers-modal", is_open=False, size="sm",
                                        ),
                                    ],
                                    className="d-flex align-items-center gap-1",
                                    style={"paddingBottom": "4px", "marginBottom": "4px"},
                                ),
                                dbc.Tabs(
                                    id="layer-tabs",
                                    active_tab=None,
                                    style={"marginBottom": "4px"},
                                ),
                                dcc.Graph(
                                    id="scatter",
                                    figure={
                                        "data": [
                                            {
                                                "mode": "lines",
                                                "type": "scatter",
                                                "x": [],
                                                "y": [],
                                            }
                                        ],
                                        "layout": {
                                            "template": pio.templates["plotly"],
                                            "uirevision": "no_change",
                                            "xaxis": dict(title="Number of Channels"),
                                            "yaxis": dict(title="Integration Gain (dB)"),
                                        },
                                    },
                                    config={"responsive": True},
                                    style={"flex": "1 1 0", "minHeight": "0"},
                                ),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Legend"),
                                        dbc.Input(
                                            id="legend",
                                            placeholder="Figure legend",
                                        ),
                                    ],
                                    className="mt-2",
                                ),
                            ],
                    className="p-2 d-flex flex-column h-100",
                ),
                className="chart-card h-100 p-2",
            ),
            width=9,
        ),
    ],
    className="g-3 panels-row mx-1 mb-1",
)


def get_app_layout():
    """
    Get the main layout for the radar coverage map application.

    Returns:
        dbc.Container: A Dash Bootstrap container that defines the complete UI layout,
            including:
            - Data stores for configuration and figures
            - Main plotting card with controls
            - Interactive elements like sliders and dropdowns
            - Download component for exporting data
            - Version information footer

    The layout is responsive and uses Bootstrap's fluid container system.
    """
    saved_sensor = None
    if os.path.exists("./settings.json"):
        try:
            with open("./settings.json", "r", encoding="utf-8") as f:
                _s = json.load(f)
            saved_sensor = _s.get("sensor")
        except (json.JSONDecodeError, OSError):
            pass

    return dbc.Container(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="config"),
            dcc.Store(id="layers-store", data=[]),
            dcc.Store(id="active-layer-store", data=None),
            dcc.Store(id="sensor-store", data=saved_sensor),
            dcc.Store(id="app-settings"),
            dcc.Download(id="download"),
            plot_card,
            dcc.Markdown(
                "v5.1 | By [Zhengyu Peng](mailto:zhengyu.peng@aptiv.com)",
                className="footer-text px-2",
            ),
        ],
        fluid=True,
        className="dbc",
        style={"height": "100vh", "display": "flex", "flexDirection": "column", "overflow": "hidden", "padding": "8px"},
    )
