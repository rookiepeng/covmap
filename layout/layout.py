"""
    Copyright (C) 2023 - PRESENT  Zhengyu Peng
    
    This module defines the layout and UI components for a radar coverage mapping application.
    It creates a responsive web interface using Dash and Bootstrap components.
"""

from dash import dcc
from dash import html

import dash_bootstrap_components as dbc

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
    ],
    value=[],
    id="flip-checklist",
    inline=True,
)

# Target RCS (Radar Cross Section) control
# Allows adjustment of target RCS from -30 to 30 dBsm
rcs_slider = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("RCS"),
                dbc.Input(
                    id="rcs-input", type="number", value=10, min=-30, max=30, step=0.1
                ),
                dbc.InputGroupText("dBsm"),
                dbc.Tooltip(
                    "Target's RCS",
                    target="rcs-input",
                    placement="top",
                ),
            ],
            size="sm",
            className="mb-1",
        ),
        dcc.Slider(
            id="rcs",
            min=-30,
            max=30,
            step=0.1,
            value=10,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "left"},
        ),
    ],
)

# Fascia loss control
# Accounts for signal loss due to vehicle fascia/radome
fascia_slider = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("Fascia"),
                dbc.Input(
                    id="fascia-input", type="number", value=0, min=-10, max=0, step=0.1
                ),
                dbc.InputGroupText("dB"),
                dbc.Tooltip(
                    "Fascia loss",
                    target="fascia-input",
                    placement="top",
                ),
            ],
            size="sm",
            className="mb-1",
        ),
        dcc.Slider(
            id="fascia",
            min=-10,
            max=0,
            step=0.1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "left"},
        ),
    ],
)

# Manufacturing margin control
# Accounts for manufacturing variations and tolerances
mfg_slider = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("MFG"),
                dbc.Input(
                    id="mfg-input", type="number", value=0, min=-10, max=0, step=0.1
                ),
                dbc.InputGroupText("dB"),
                dbc.Tooltip(
                    "Manufacturer margin",
                    target="mfg-input",
                    placement="top",
                ),
            ],
            size="sm",
            className="mb-1",
        ),
        dcc.Slider(
            id="mfg",
            min=-10,
            max=0,
            step=0.1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "left"},
        ),
    ],
)

# Temperature-related loss control
# Accounts for performance variations due to temperature
temp_slider = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("Temp"),
                dbc.Input(
                    id="temp-input", type="number", min=-10, max=0, step=0.1, value=0
                ),
                dbc.InputGroupText("dB"),
                dbc.Tooltip(
                    "Temperature loss",
                    target="temp-input",
                    placement="top",
                ),
            ],
            size="sm",
            className="mb-1",
        ),
        dcc.Slider(
            id="temp",
            min=-10,
            max=0,
            step=0.1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "left"},
        ),
    ],
)

# Rain-induced loss control
# Accounts for signal attenuation due to rain
rain_slider = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("Rain"),
                dbc.Input(
                    id="rain-input", type="number", min=-10, max=0, step=0.1, value=0
                ),
                dbc.InputGroupText("dB"),
                dbc.Tooltip(
                    "Rain damping loss",
                    target="rain-input",
                    placement="top",
                ),
            ],
            size="sm",
            className="mb-1",
        ),
        dcc.Slider(
            id="rain",
            min=-10,
            max=0,
            step=0.1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "left"},
        ),
    ],
)

# Elevation misalignment control
# Allows adjustment for mounting angle errors
misalign_slider = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("Misalgn"),
                dbc.Input(
                    id="misalign-input", type="number", min=-20, max=20, step=1, value=0
                ),
                dbc.InputGroupText("deg"),
                dbc.Tooltip(
                    "Elevation misalignment",
                    target="misalign-input",
                    placement="top",
                ),
            ],
            size="sm",
            className="mb-1",
        ),
        dcc.Slider(
            id="misalign",
            min=-20,
            max=20,
            step=1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "left"},
        ),
    ],
)

# Azimuth offset control
# Allows adjustment of azimuth angle offset
az_slider = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("Az offset"),
                dbc.Input(
                    id="az-offset-input",
                    type="number",
                    min=-180,
                    max=180,
                    step=1,
                    value=0,
                ),
                dbc.InputGroupText("deg"),
                dbc.Tooltip(
                    "Azimuth offset",
                    target="az-offset-input",
                    placement="top",
                ),
            ],
            size="sm",
            className="mb-1",
        ),
        dcc.Slider(
            id="az-offset",
            min=-180,
            max=180,
            step=1,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "left"},
        ),
    ],
)

# Longitudinal position offset control
# Adjusts sensor position along vehicle's length
long_slider = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("Long"),
                dbc.Input(
                    id="long-input", type="number", min=-5, max=5, step=0.001, value=0
                ),
                dbc.InputGroupText("m"),
                dbc.Tooltip(
                    "Longitudinal offset",
                    target="long-input",
                    placement="top",
                ),
            ],
            size="sm",
            className="mb-1",
        ),
        dcc.Slider(
            id="long",
            min=-5,
            max=5,
            step=0.001,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "left"},
        ),
    ],
)

# Lateral position offset control
# Adjusts sensor position across vehicle's width
lat_slider = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("Lat"),
                dbc.Input(
                    id="lat-input", type="number", min=-5, max=5, step=0.001, value=0
                ),
                dbc.InputGroupText("m"),
                dbc.Tooltip(
                    "Latitudinal offset",
                    target="lat-input",
                    placement="top",
                ),
            ],
            size="sm",
            className="mb-1",
        ),
        dcc.Slider(
            id="lat",
            min=-5,
            max=5,
            step=0.001,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "left"},
        ),
    ],
)

# Height offset control
# Adjusts sensor mounting height
height_slider = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("Height"),
                dbc.Input(
                    id="height-input", type="number", min=-5, max=5, step=0.001, value=0
                ),
                dbc.InputGroupText("m"),
                dbc.Tooltip(
                    "Height offset",
                    target="height-input",
                    placement="top",
                ),
            ],
            size="sm",
            className="mb-1",
        ),
        dcc.Slider(
            id="height",
            min=-5,
            max=5,
            step=0.001,
            value=0,
            marks=None,
            updatemode="drag",
            tooltip={"always_visible": False, "placement": "left"},
        ),
    ],
)

# Main plotting card containing all UI controls and the plot
plot_card = dbc.Card(
    [
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Row(
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
                                        className="mb-3",
                                    ),
                                    html.Div(
                                        [
                                            dbc.Row(
                                                id="property-container", children=[]
                                            ),
                                            dbc.Col(html.Hr(), width=12),
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
                                                className="mb-3",
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
                                                className="mb-3",
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
                                                className="mb-3",
                                            ),
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
                                            ),
                                            flip_checklist,
                                            dbc.Col(html.Hr()),
                                            dbc.Form(
                                                [
                                                    rcs_slider,
                                                    fascia_slider,
                                                    mfg_slider,
                                                    temp_slider,
                                                    rain_slider,
                                                    misalign_slider,
                                                    az_slider,
                                                    long_slider,
                                                    lat_slider,
                                                    height_slider,
                                                ]
                                            ),
                                        ],
                                        style={
                                            "overflow-y": "scroll",
                                            "height": "82vh",
                                        },
                                    ),
                                ]
                            ),
                            width=3,
                        ),
                        dbc.Col(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button(
                                                    "Clear last held plot",
                                                    id="clear-last-plot",
                                                    n_clicks=0,
                                                    color="warning",
                                                    class_name="my-1",
                                                ),
                                                dbc.Button(
                                                    "Clear all held plots",
                                                    id="clear-plot",
                                                    n_clicks=0,
                                                    color="danger",
                                                    class_name="my-1",
                                                ),
                                            ],
                                            style={"float": "right"},
                                        ),
                                        width={"size": 6, "offset": 6},
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
                                                "xaxis": dict(
                                                    title="Number of Channels"
                                                ),
                                                "yaxis": dict(
                                                    title="Integration Gain (dB)"
                                                ),
                                            },
                                        },
                                        style={"height": "76vh"},
                                    ),
                                    dbc.Col(html.Hr()),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Legend"),
                                            dbc.Input(
                                                id="legend",
                                                placeholder="Figure legend",
                                                type="text",
                                            ),
                                            dbc.Button(
                                                "Hold plot",
                                                id="hold-plot",
                                                color="success",
                                                n_clicks=0,
                                            ),
                                            dbc.DropdownMenu(
                                                export_menu_items,
                                                label="Export",
                                                color="info",
                                            ),
                                        ],
                                        className="mt-3",
                                    ),
                                ]
                            ),
                            width=9,
                        ),
                    ],
                    class_name="g-5",
                )
            ],
            class_name="mx-3 my-3",
        ),
    ],
    class_name="mt-2 mb-2",
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
    return dbc.Container(
        [
            dcc.Store(id="config"),
            dcc.Store(id="figure-data", data=[]),
            dcc.Store(id="new-figure-data", data=[]),
            dcc.Store(id="sensor-store", data=[], storage_type="local"),
            dcc.Download(id="download"),
            plot_card,
            # html.Hr(),
            dcc.Markdown(
                "v4.3 | By [Zhengyu Peng](mailto:zhengyu.peng@aptiv.com)",
                className="pb-1",
            ),
        ],
        fluid=True,
        className="dbc_light",
    )
