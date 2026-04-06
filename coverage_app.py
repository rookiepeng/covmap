"""
Copyright (C) 2023 - PRESENT  Zhengyu Peng

Coverage Map Application - A tool for visualizing radar coverage patterns.
"""

from typing import Dict, List, Any, Tuple, Union, Optional

import json
import os
import base64

import dash
from dash import dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

import numpy as np
import plotly.graph_objects as go

import pandas as pds

from flaskwebgui import FlaskUI

from roc.tools import roc_snr

from layout.layout import get_app_layout

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


def load_config(json_file: str) -> Dict[str, Any]:
    """
    Load and parse a radar configuration from a JSON file.

    Parameters
    ----------
    json_file : str
        Path to the JSON configuration file

    Returns
    -------
    dict
        Parsed configuration data containing radar parameters

    Raises
    ------
    JSONDecodeError
        If the file contains invalid JSON
    FileNotFoundError
        If the specified file does not exist
    """
    with open(json_file, "r", encoding="utf-8") as read_file:
        return json.load(read_file)


@app.callback(
    output={
        "sensor_opts": Output("sensor", "options"),
        "sensor_val": Output("sensor", "value"),
    },
    inputs={
        "unused_btn": Input("reload", "n_clicks"),
    },
    state={
        "sensor_store": State("sensor-store", "data"),
    },
)
def reload(unused_btn: Any, sensor_store: str) -> Dict[str, Any]:
    """
    Reload the list of available radar sensor configurations.

    Scans the ./radar directory for .json configuration files and updates
    the sensor selection dropdown.

    Parameters
    ----------
    unused_btn : Any
        Unused click event parameter
    sensor_store : str
        Currently selected sensor value

    Returns
    -------
    dict
        Dictionary containing:
            sensor_opts: List of available sensor options
            sensor_val: Currently selected sensor value

    Notes
    -----
    The function maintains the currently selected sensor if still available,
    otherwise defaults to the first sensor found.
    """

    radar_list = []
    json_list = []
    for unused_dirpath, unused_dirnames, files in os.walk("./radar"):
        for name in files:
            if name.lower().endswith(".json"):
                json_list.append(name)
                radar_list.append({"label": name, "value": name})

    if sensor_store in json_list:
        sensor_val = sensor_store
    else:
        sensor_val = radar_list[0]["value"]

    return {"sensor_opts": radar_list, "sensor_val": sensor_val}


@app.callback(
    output={
        "reload": Output("reload", "n_clicks"),
        "sensor_store": Output("sensor-store", "data", allow_duplicate=True),
    },
    inputs={"list_of_contents": Input("upload-config", "contents")},
    state={
        "list_of_names": State("upload-config", "filename"),
        "unused_list_of_dates": State("upload-config", "last_modified"),
        "n_clicks": State("reload", "n_clicks"),
    },
    prevent_initial_call=True,
)
def upload_config(
    list_of_contents: Optional[str],
    list_of_names: str,
    unused_list_of_dates: Any,
    n_clicks: int,
) -> Dict[str, Union[int, str]]:
    """
    Process and save an uploaded radar configuration file.

    Parameters
    ----------
    list_of_contents : str
        Base64 encoded contents of the uploaded file
    list_of_names : str
        Name of the uploaded file
    unused_list_of_dates : Any
        Unused parameter for file modification dates
    n_clicks : int
        Number of times the upload button has been clicked

    Returns
    -------
    dict
        Dictionary containing:
            reload: Updated click counter
            sensor_store: Name of the saved configuration file

    Raises
    ------
    PreventUpdate
        If no file contents are provided
    JSONDecodeError
        If the uploaded file contains invalid JSON
    """

    if list_of_contents is None:
        raise PreventUpdate

    decoded_str = base64.b64decode(list_of_contents.split("base64,")[1])
    config = json.loads(decoded_str)

    with open("./radar/" + list_of_names, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    return {"reload": n_clicks + 1, "sensor_store": list_of_names}


@app.callback(
    output={
        "config": Output("config", "data"),
        "misalign_min": Output("misalign", "min"),
        "misalign_max": Output("misalign", "max"),
        "misalign": Output("misalign", "value", allow_duplicate=True),
        "sensor_store": Output("sensor-store", "data"),
    },
    inputs={
        "sensor": Input("sensor", "value"),
    },
    state={
        "misalgin_state": State("misalign", "value"),
    },
    prevent_initial_call=True,
)
def sensor_select(sensor: str, misalgin_state: float) -> Dict[str, Any]:
    """
    Configure display parameters based on selected radar sensor.

    Loads the sensor configuration and sets up misalignment ranges and values.

    Parameters
    ----------
    sensor : str
        Name of the selected sensor configuration file
    misalgin_state : float
        Current misalignment angle value

    Returns
    -------
    dict
        Dictionary containing updated sensor configuration and UI parameters:
            config: Full sensor configuration
            misalign_min/max: Misalignment angle limits
            misalign: Current misalignment value
            misalign_input_min/max: Input field limits
            misalign_input: Current input value
            sensor_store: Selected sensor name
    """

    config = load_config("./radar/" + sensor)
    misalign_min = config["el_fov"][0]
    misalign_max = config["el_fov"][1]

    if misalgin_state < misalign_min or misalgin_state > misalign_max:
        misalign = 0
    else:
        misalign = misalgin_state
    return {
        "config": config,
        "misalign_min": misalign_min,
        "misalign_max": misalign_max,
        "misalign": misalign,
        "sensor_store": sensor,
    }


@app.callback(
    output={
        "fig_data": Output("figure-data", "data", allow_duplicate=True),
    },
    inputs={
        "unused_plot_type": Input("plot", "value"),
        "unused_clear_btn": Input("clear-plot", "n_clicks"),
    },
    prevent_initial_call=True,
)
def clear_plot(unused_clear_btn: Any, unused_plot_type: Any) -> Dict[str, List]:
    """
    Clear all plots from the display.

    Parameters
    ----------
    unused_clear_btn : Any
        Unused clear button click event
    unused_plot_type : Any
        Unused plot type parameter

    Returns
    -------
    dict
        Empty figure data dictionary
    """

    return {"fig_data": []}


@app.callback(
    output={
        "fig_data": Output("figure-data", "data", allow_duplicate=True),
    },
    inputs={
        "unused_clear_btn": Input("clear-last-plot", "n_clicks"),
    },
    state={
        "fig_data_input": State("figure-data", "data"),
    },
    prevent_initial_call=True,
)
def clear_last_plot(unused_clear_btn: Any, fig_data_input: List) -> Dict[str, List]:
    """
    Remove the most recently added plot from the display.

    Parameters
    ----------
    unused_clear_btn : Any
        Unused clear button click event
    fig_data_input : list
        Current list of plot data

    Returns
    -------
    dict
        Updated figure data with last plot removed

    Raises
    ------
    PreventUpdate
        If there are no plots to remove
    """

    if len(fig_data_input) > 0:
        fig_data_input.pop(-1)
        return {"fig_data": fig_data_input}

    raise PreventUpdate


@app.callback(
    output={
        "fig_data": Output("figure-data", "data"),
    },
    inputs={
        "unused_hold_btn": Input("hold-plot", "n_clicks"),
    },
    state={
        "current_figs": State("figure-data", "data"),
        "new_fig": State("new-figure-data", "data"),
    },
    prevent_initial_call=True,
)
def hold_plot(
    unused_hold_btn: Any, current_figs: List, new_fig: List
) -> Dict[str, List]:
    """
    Add a new plot while maintaining existing plots.

    Parameters
    ----------
    unused_hold_btn : Any
        Unused hold button click event
    current_figs : list
        List of existing plot data
    new_fig : list
        New plot data to add

    Returns
    -------
    dict
        Combined figure data including new plot
    """

    return {"fig_data": current_figs + new_fig}


@app.callback(
    output={
        "fig": Output("scatter", "figure"),
        "new_fig": Output("new-figure-data", "data"),
        "property_container": Output("property-container", "children"),
        "legend_entry": Output("legend", "value"),
    },
    inputs={
        "pd": Input("pd", "value"),
        "pfa": Input("pfa", "value"),
        "rcs": Input("rcs", "value"),
        "fascia_loss": Input("fascia", "value"),
        "mfg_loss": Input("mfg", "value"),
        "temp_loss": Input("temp", "value"),
        "rain_loss": Input("rain", "value"),
        "vert_misalign_angle": Input("misalign", "value"),
        "roll_offset": Input("roll-offset", "value"),
        "az_offset": Input("az-offset", "value"),
        "sw_model": Input("integration", "value"),
        "plot_type": Input("plot", "value"),
        "fig_data": Input("figure-data", "data"),
        "new_legend_entry": Input("legend", "value"),
        "flip": Input("flip-checklist", "value"),
        "inset_position": Input("inset-position", "value"),
        "long_offset": Input("long", "value"),
        "lat_offset": Input("lat", "value"),
        "height_offset": Input("height", "value"),
        "config": Input("config", "data"),
    },
    state={
        "min_pd": State("pd", "min"),
        "max_pd": State("pd", "max"),
        "min_pfa": State("pfa", "min"),
        "max_pfa": State("pfa", "max"),
    },
    prevent_initial_call=True,
)
def coverage_plot(
    pd: float,
    pfa: float,
    rcs: float,
    fascia_loss: float,
    mfg_loss: float,
    temp_loss: float,
    rain_loss: float,
    vert_misalign_angle: float,
    roll_offset: float,
    az_offset: float,
    sw_model: str,
    plot_type: str,
    new_legend_entry: str,
    min_pd: float,
    max_pd: float,
    min_pfa: float,
    max_pfa: float,
    fig_data: List,
    flip: List[str],
    inset_position: str,
    long_offset: float,
    lat_offset: float,
    height_offset: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a coverage plot based on the input parameters.

    Parameters
    ----------
    pd : float
        Probability of detection
    pfa : float
        Probability of false alarm
    rcs : float
        Radar cross section
    fascia_loss : float
        Fascia loss
    mfg_loss : float
        Manufacturing loss
    temp_loss : float
        Temperature loss
    rain_loss : float
        Rain loss
    vert_misalign_angle : float
        Vertical misalignment angle
    roll_offset : float
        Roll offset around boresight axis
    az_offset : float
        Azimuth offset
    sw_model : str
        Software model
    plot_type : str
        Type of coverage plot
    new_legend_entry : str
        New legend entry
    min_pd : float
        Minimum probability of detection
    max_pd : float
        Maximum probability of detection
    min_pfa : float
        Minimum probability of false alarm
    max_pfa : float
        Maximum probability of false alarm
    fig_data : list
        Existing figure data
    flip : list
        Flip patterns
    long_offset : float
        Longitude offset
    lat_offset : float
        Latitude offset
    height_offset : float
        Height offset
    config : dict
        Configuration parameters

    Returns
    -------
    dict
        Dictionary containing the updated figure data,
        figure layout, property container, and legend entry
    """

    if pd is None:
        raise PreventUpdate
    if pd < min_pd or pd > max_pd:
        raise PreventUpdate

    if pfa is None:
        raise PreventUpdate
    if pfa < min_pfa or pfa > max_pfa:
        raise PreventUpdate

    if rcs is None:
        raise PreventUpdate
    if fascia_loss is None:
        raise PreventUpdate
    if mfg_loss is None:
        raise PreventUpdate
    if temp_loss is None:
        raise PreventUpdate
    if rain_loss is None:
        raise PreventUpdate
    if vert_misalign_angle is None:
        raise PreventUpdate
    if roll_offset is None:
        raise PreventUpdate
    if az_offset is None:
        raise PreventUpdate
    if long_offset is None:
        raise PreventUpdate
    if lat_offset is None:
        raise PreventUpdate
    if height_offset is None:
        raise PreventUpdate

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # check if the patterns need to be flipped
    flip_az = False
    flip_el = False
    if "flip_az" in flip:
        flip_az = True
    if "flip_el" in flip:
        flip_el = True

    if flip_az:
        az_start = -config["az_fov"][1]
        az_end = -config["az_fov"][0]
    else:
        az_start = config["az_fov"][0]
        az_end = config["az_fov"][1]

    if flip_el:
        el_start = -config["el_fov"][1]
        el_end = -config["el_fov"][0]
    else:
        el_start = config["el_fov"][0]
        el_end = config["el_fov"][1]

    chamber_snr = config["chamber_snr"]
    chamber_rcs = config["chamber_rcs"]
    chamber_range = config["chamber_range"]
    chamber_az = config.get("chamber_az", 0)
    chamber_el = config.get("chamber_el", 0)

    straddling_loss = config["straddling_loss"]
    channels = config["num_channels"]

    discription = config.get("discription", None)
    extra_gain_loss = config.get("extra_proc_gain_loss", 0)

    # normalize azimuth pattern based on chamber angle
    az_ptn_full = np.array(config["az_ptn"])
    az_ang_full = np.array(config["az_ang"])
    az_ptn_full = az_ptn_full - az_ptn_full[np.where(az_ang_full == chamber_az)]
    if flip_az:
        az_ptn_full = np.flip(az_ptn_full)
        az_ang_full = np.flip(-az_ang_full)

    # Trimmed arrays with edge clamping for sweep/display
    idx = np.where(
        np.logical_and(az_ang_full >= az_start - 1, az_ang_full <= az_end + 1)
    )
    az_ang = az_ang_full[idx]
    az_ptn = az_ptn_full[idx].copy()
    az_ptn[0] = -1000
    az_ptn[-1] = -1000

    # normalize elevation pattern based on chamber angle
    el_ptn_full = np.array(config["el_ptn"])
    el_ang_full = np.array(config["el_ang"])
    el_ptn_full = el_ptn_full - el_ptn_full[np.where(el_ang_full == chamber_el)]
    if flip_el:
        el_ptn_full = np.flip(el_ptn_full)
        el_ang_full = np.flip(-el_ang_full)

    # Trimmed arrays with edge clamping for sweep/display
    idx = np.where(
        np.logical_and(el_ang_full >= el_start - 1, el_ang_full <= el_end + 1)
    )
    el_ang = el_ang_full[idx]
    el_ptn = el_ptn_full[idx].copy()
    el_ptn[0] = -1000
    el_ptn[-1] = -1000

    min_snr = roc_snr(pfa, pd, 1, sw_model)
    nci_gain = roc_snr(pfa, pd, 1, sw_model) - roc_snr(pfa, pd, channels, sw_model)

    roll_rad = roll_offset / 180 * np.pi

    # Compute misalignment loss at boresight for display purposes
    az_at_boresight = vert_misalign_angle * np.sin(roll_rad)
    el_at_boresight = vert_misalign_angle * np.cos(roll_rad)
    az_loss_at_boresight = np.interp(
        az_at_boresight, az_ang_full, az_ptn_full, left=-1000, right=-1000
    )
    el_loss_at_boresight = np.interp(
        el_at_boresight, el_ang_full, el_ptn_full, left=-1000, right=-1000
    )
    el_missalign_loss = az_loss_at_boresight + el_loss_at_boresight

    max_range = 10 ** (
        (
            chamber_snr
            + 40 * np.log10(chamber_range)
            - chamber_rcs
            + rcs
            + nci_gain
            + fascia_loss
            + mfg_loss
            + temp_loss
            + rain_loss
            + straddling_loss
            + extra_gain_loss
            - min_snr
        )
        / 40
    )

    if trigger_id == "legend":
        legend = new_legend_entry
    else:
        legend = (
            str(rcs)
            + " dBsm, "
            + sw_model
            + ",<br>"
            + str(abs(fascia_loss))
            + " dB Fascia Loss,<br>"
            + str(abs(temp_loss))
            + " dB Temp Loss,<br>"
            + str(abs(rain_loss))
            + " dB Rain Loss,<br>"
            + str(abs(mfg_loss))
            + " dB MFG,<br>"
            + str(vert_misalign_angle)
            + " deg Misalignment<br>"
        )

    # clear all the held plots if the plot type is changed
    if trigger_id == "plot":
        fig_data = []

    # Compute 2D az-el range heatmap for inset
    az_grid, el_grid = np.meshgrid(az_ang, el_ang)
    az_antenna_2d = az_grid * np.cos(roll_rad) + el_grid * np.sin(roll_rad)
    el_antenna_2d = -az_grid * np.sin(roll_rad) + el_grid * np.cos(roll_rad)
    az_loss_2d = np.interp(
        az_antenna_2d.ravel(), az_ang_full, az_ptn_full, left=-1000, right=-1000
    ).reshape(az_grid.shape)
    el_loss_2d = np.interp(
        el_antenna_2d.ravel(), el_ang_full, el_ptn_full, left=-1000, right=-1000
    ).reshape(el_grid.shape)
    combined_2d = az_loss_2d + el_loss_2d
    oov_2d = (
        (az_antenna_2d < az_start)
        | (az_antenna_2d > az_end)
        | (el_antenna_2d < el_start)
        | (el_antenna_2d > el_end)
    )
    combined_2d[oov_2d] = -1000
    range_2d = max_range * 10 ** (combined_2d / 40)
    range_2d[range_2d < 1e-6] = np.nan

    # Inset position domains
    inset_domains = {
        "top-right": {"x": [0.72, 0.98], "y": [0.67, 0.95]},
        "top-left": {"x": [0.04, 0.30], "y": [0.67, 0.95]},
        "bottom-right": {"x": [0.72, 0.98], "y": [0.05, 0.33]},
        "bottom-left": {"x": [0.04, 0.30], "y": [0.05, 0.33]},
    }
    show_inset = inset_position != "hidden" and inset_position in inset_domains
    if show_inset:
        inset_dom = inset_domains[inset_position]
    else:
        inset_dom = {"x": [0.04, 0.30], "y": [0.67, 0.95]}

    inset_heatmap = {
        "type": "heatmap",
        "z": range_2d.tolist(),
        "x": az_ang.tolist(),
        "y": el_ang.tolist(),
        "colorscale": "Viridis",
        "showscale": False,
        "hoverinfo": "x+y+z",
        "xaxis": "x2",
        "yaxis": "y2",
    }

    if plot_type == "Azimuth Coverage":
        # With roll offset, compute antenna-frame angles for each ground azimuth
        az_antenna = az_ang * np.cos(roll_rad) + vert_misalign_angle * np.sin(roll_rad)
        el_antenna = -az_ang * np.sin(roll_rad) + vert_misalign_angle * np.cos(roll_rad)
        az_loss = np.interp(
            az_antenna, az_ang_full, az_ptn_full, left=-1000, right=-1000
        )
        el_loss = np.interp(
            el_antenna, el_ang_full, el_ptn_full, left=-1000, right=-1000
        )
        combined_ptn = az_loss + el_loss
        # Limit to FOV
        out_of_fov = (
            (az_antenna < az_start)
            | (az_antenna > az_end)
            | (el_antenna < el_start)
            | (el_antenna > el_end)
        )
        combined_ptn[out_of_fov] = -1000
        coverage = max_range * 10 ** (combined_ptn / 40)
        coverage_long = (
            coverage * np.cos((az_ang + az_offset) / 180 * np.pi) + long_offset
        )
        coverage_lat = (
            coverage * np.sin((az_ang + az_offset) / 180 * np.pi) + lat_offset
        )
        new_fig = [
            {
                "mode": "lines",
                "type": "scatter",
                "x": coverage_long,
                "y": coverage_lat,
                "fill": "tozeroy",
                "name": legend,
            }
        ]
        # Cut line for inset: horizontal line at el = misalign across az range
        inset_cut = {
            "type": "scatter",
            "mode": "lines",
            "x": [float(az_ang[0]), float(az_ang[-1])],
            "y": [float(vert_misalign_angle), float(vert_misalign_angle)],
            "line": {"color": "red", "width": 2},
            "showlegend": False,
            "hoverinfo": "skip",
            "xaxis": "x2",
            "yaxis": "y2",
        }
        new_fig.extend([inset_heatmap, inset_cut] if show_inset else [])
        fig_layout = {
            "template": "seaborn",
            "margin": {"l": 60, "r": 10, "t": 30, "b": 50},
            "xaxis": {"title": {"text": "Longitude (m)"}, "domain": [0, 1]},
            "yaxis": {
                "title": {"text": "Latitude (m)"},
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            "xaxis2": {
                "domain": inset_dom["x"],
                "anchor": "y2",
                "showgrid": False,
                "visible": show_inset,
                "showline": True,
                "mirror": True,
                "tickfont": {"size": 9},
            },
            "yaxis2": {
                "domain": inset_dom["y"],
                "anchor": "x2",
                "showgrid": False,
                "visible": show_inset,
                "showline": True,
                "mirror": True,
                "tickfont": {"size": 9},
            },
        }
    elif plot_type == "Azimuth vs. Range":
        az_antenna = az_ang * np.cos(roll_rad) + vert_misalign_angle * np.sin(roll_rad)
        el_antenna = -az_ang * np.sin(roll_rad) + vert_misalign_angle * np.cos(roll_rad)
        az_loss = np.interp(
            az_antenna, az_ang_full, az_ptn_full, left=-1000, right=-1000
        )
        el_loss = np.interp(
            el_antenna, el_ang_full, el_ptn_full, left=-1000, right=-1000
        )
        combined_ptn = az_loss + el_loss
        out_of_fov = (
            (az_antenna < az_start)
            | (az_antenna > az_end)
            | (el_antenna < el_start)
            | (el_antenna > el_end)
        )
        combined_ptn[out_of_fov] = -1000
        coverage = max_range * 10 ** (combined_ptn / 40)
        new_fig = [
            {
                "mode": "lines",
                "type": "scatter",
                "x": az_ang,
                "y": coverage,
                "fill": "tozeroy",
                "name": legend,
            }
        ]
        inset_cut = {
            "type": "scatter",
            "mode": "lines",
            "x": [float(az_ang[0]), float(az_ang[-1])],
            "y": [float(vert_misalign_angle), float(vert_misalign_angle)],
            "line": {"color": "red", "width": 2},
            "showlegend": False,
            "hoverinfo": "skip",
            "xaxis": "x2",
            "yaxis": "y2",
        }
        new_fig.extend([inset_heatmap, inset_cut] if show_inset else [])
        fig_layout = {
            "template": "seaborn",
            "margin": {"l": 60, "r": 10, "t": 30, "b": 50},
            "xaxis": {"title": {"text": "Azimuth (deg)"}, "domain": [0, 1]},
            "yaxis": {"title": {"text": "Range (m)"}},
            "xaxis2": {
                "domain": inset_dom["x"],
                "anchor": "y2",
                "showgrid": False,
                "visible": show_inset,
                "showline": True,
                "mirror": True,
                "tickfont": {"size": 9},
            },
            "yaxis2": {
                "domain": inset_dom["y"],
                "anchor": "x2",
                "showgrid": False,
                "visible": show_inset,
                "showline": True,
                "mirror": True,
                "tickfont": {"size": 9},
            },
        }
    elif plot_type == "Elevation Coverage":
        # With roll offset, compute antenna-frame angles for each ground elevation
        az_antenna_el = az_offset * np.cos(roll_rad) + el_ang * np.sin(roll_rad)
        el_antenna_el = -az_offset * np.sin(roll_rad) + el_ang * np.cos(roll_rad)
        az_loss_el = np.interp(
            az_antenna_el, az_ang_full, az_ptn_full, left=-1000, right=-1000
        )
        el_loss_el = np.interp(
            el_antenna_el, el_ang_full, el_ptn_full, left=-1000, right=-1000
        )
        combined_ptn_el = az_loss_el + el_loss_el
        out_of_fov = (
            (az_antenna_el < az_start)
            | (az_antenna_el > az_end)
            | (el_antenna_el < el_start)
            | (el_antenna_el > el_end)
        )
        combined_ptn_el[out_of_fov] = -1000
        coverage = max_range * 10 ** (combined_ptn_el / 40)
        coverage_long = (
            coverage * np.cos((el_ang + vert_misalign_angle) / 180 * np.pi)
            + long_offset
        )
        coverage_height = (
            coverage * np.sin((el_ang + vert_misalign_angle) / 180 * np.pi)
            + height_offset
        )
        new_fig = [
            {
                "mode": "lines",
                "type": "scatter",
                "x": coverage_long,
                "y": coverage_height,
                "fill": "tozeroy",
                "name": legend,
            }
        ]
        # Cut line for inset: vertical line at az = az_offset across el range
        inset_cut = {
            "type": "scatter",
            "mode": "lines",
            "x": [float(az_offset), float(az_offset)],
            "y": [float(el_ang[0]), float(el_ang[-1])],
            "line": {"color": "red", "width": 2},
            "showlegend": False,
            "hoverinfo": "skip",
            "xaxis": "x2",
            "yaxis": "y2",
        }
        new_fig.extend([inset_heatmap, inset_cut] if show_inset else [])
        fig_layout = {
            "template": "seaborn",
            "margin": {"l": 60, "r": 10, "t": 30, "b": 50},
            "xaxis": {"title": {"text": "Longitude (m)"}, "domain": [0, 1]},
            "yaxis": {
                "title": {"text": "Height (m)"},
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            "xaxis2": {
                "domain": inset_dom["x"],
                "anchor": "y2",
                "showgrid": False,
                "visible": show_inset,
                "showline": True,
                "mirror": True,
                "tickfont": {"size": 9},
            },
            "yaxis2": {
                "domain": inset_dom["y"],
                "anchor": "x2",
                "showgrid": False,
                "visible": show_inset,
                "showline": True,
                "mirror": True,
                "tickfont": {"size": 9},
            },
        }
    elif plot_type == "Elevation vs. Range":
        az_antenna_el = az_offset * np.cos(roll_rad) + el_ang * np.sin(roll_rad)
        el_antenna_el = -az_offset * np.sin(roll_rad) + el_ang * np.cos(roll_rad)
        az_loss_el = np.interp(
            az_antenna_el, az_ang_full, az_ptn_full, left=-1000, right=-1000
        )
        el_loss_el = np.interp(
            el_antenna_el, el_ang_full, el_ptn_full, left=-1000, right=-1000
        )
        combined_ptn_el = az_loss_el + el_loss_el
        out_of_fov = (
            (az_antenna_el < az_start)
            | (az_antenna_el > az_end)
            | (el_antenna_el < el_start)
            | (el_antenna_el > el_end)
        )
        combined_ptn_el[out_of_fov] = -1000
        coverage = max_range * 10 ** (combined_ptn_el / 40)
        new_fig = [
            {
                "mode": "lines",
                "type": "scatter",
                "x": el_ang,
                "y": coverage,
                "fill": "tozeroy",
                "name": legend,
            }
        ]
        inset_cut = {
            "type": "scatter",
            "mode": "lines",
            "x": [float(az_offset), float(az_offset)],
            "y": [float(el_ang[0]), float(el_ang[-1])],
            "line": {"color": "red", "width": 2},
            "showlegend": False,
            "hoverinfo": "skip",
            "xaxis": "x2",
            "yaxis": "y2",
        }
        new_fig.extend([inset_heatmap, inset_cut] if show_inset else [])
        fig_layout = {
            "template": "seaborn",
            "margin": {"l": 60, "r": 10, "t": 30, "b": 50},
            "xaxis": {"title": {"text": "Elevation (deg)"}, "domain": [0, 1]},
            "yaxis": {"title": {"text": "Range (m)"}},
            "xaxis2": {
                "domain": inset_dom["x"],
                "anchor": "y2",
                "showgrid": False,
                "visible": show_inset,
                "showline": True,
                "mirror": True,
                "tickfont": {"size": 9},
            },
            "yaxis2": {
                "domain": inset_dom["y"],
                "anchor": "x2",
                "showgrid": False,
                "visible": show_inset,
                "showline": True,
                "mirror": True,
                "tickfont": {"size": 9},
            },
        }

    container = []

    if discription is not None:
        container.append(
            dbc.FormText(
                dcc.Markdown(discription, dangerously_allow_html=True), color="primary"
            )
        )

    container.append(
        dbc.FormText(
            "Chamber SNR: "
            + str(chamber_snr)
            + " dB ("
            + str(chamber_rcs)
            + " dBsm at "
            + str(chamber_range)
            + " m)"
        )
    )
    container.append(dbc.FormText("Number of NCI Channels: " + str(channels)))
    container.append(dbc.FormText("Min SNR: " + str(round(min_snr, 3)) + " dB"))
    container.append(
        dbc.FormText("Integration Gain: " + str(round(nci_gain, 3)) + " dB")
    )
    container.append(
        dbc.FormText("Straddling Loss: " + str(round(straddling_loss, 3)) + " dB")
    )
    container.append(
        dbc.FormText("Misalignment Loss: " + str(round(el_missalign_loss, 3)) + " dB")
    )
    container.append(
        dbc.FormText(
            "Extra Processing Gain/Loss: " + str(round(extra_gain_loss, 3)) + " dB"
        )
    )

    return {
        "fig": go.Figure(
            data=fig_data + new_fig,
            layout=fig_layout,
        ).to_dict(),
        "new_fig": new_fig,
        "property_container": container,
        "legend_entry": legend,
    }


@app.callback(
    Output("download", "data", allow_duplicate=True),
    Input("export-png", "n_clicks"),
    State("scatter", "figure"),
    prevent_initial_call=True,
)
def export_png(unused_n_clicks: Any, fig: Dict[str, Any]) -> Any:
    """
    Export the current figure as a PNG file.

    Parameters
    ----------
    unused_n_clicks : Any
        Unused click event parameter
    fig : dict
        The figure object to be exported

    Returns
    -------
    dcc.send_file
        Sends the exported PNG file to the user for download
    """

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
    """
    Export the current figure as an SVG file.

    Parameters
    ----------
    unused_n_clicks : Any
        Unused click event parameter
    fig : dict
        The figure object to be exported

    Returns
    -------
    dcc.send_file
        Sends the exported SVG file to the user for download
    """

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
    """
    Export the current figure as an HTML file.

    Parameters
    ----------
    unused_n_clicks : Any
        Unused click event parameter
    fig : dict
        The figure object to be exported

    Returns
    -------
    dcc.send_file
        Sends the exported HTML file to the user for download
    """

    figure = go.Figure(fig)

    if not os.path.exists("temp"):
        os.mkdir("temp")

    figure.write_html("./temp/plot.html")
    return dcc.send_file("./temp/plot.html")


@app.callback(
    Output("download", "data", allow_duplicate=True),
    Input("export-data", "n_clicks"),
    [
        State("new-figure-data", "data"),
        State("plot", "value"),
    ],
    prevent_initial_call=True,
)
def export_data(
    unused_n_clicks: Any, data: List[Dict[str, List[float]]], plot_type: str
) -> Any:
    """
    Export the raw data as a CSV file.

    Parameters
    ----------
    unused_n_clicks : Any
        Unused click event parameter
    data : list
        The data to be exported
    plot_type : str
        The type of plot associated with the data

    Returns
    -------
    dcc.send_data_frame
        Sends the exported CSV file to the user for download
    """

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


@app.callback(
    Output("app-settings", "data"),
    inputs={
        "pd": Input("pd", "value"),
        "pfa": Input("pfa", "value"),
        "integration": Input("integration", "value"),
        "rcs": Input("rcs", "value"),
        "plot": Input("plot", "value"),
        "inset_position": Input("inset-position", "value"),
        "flip": Input("flip-checklist", "value"),
        "fascia": Input("fascia", "value"),
        "mfg": Input("mfg", "value"),
        "temp": Input("temp", "value"),
        "rain": Input("rain", "value"),
        "az_offset": Input("az-offset", "value"),
        "misalign": Input("misalign", "value"),
        "roll_offset": Input("roll-offset", "value"),
        "long_offset": Input("long", "value"),
        "lat_offset": Input("lat", "value"),
        "height_offset": Input("height", "value"),
        "sensor": Input("sensor", "value"),
    },
    prevent_initial_call=True,
)
def save_settings(
    pd: float,
    pfa: float,
    integration: str,
    rcs: float,
    plot: str,
    inset_position: str,
    flip: List[str],
    fascia: float,
    mfg: float,
    temp: float,
    rain: float,
    az_offset: float,
    misalign: float,
    roll_offset: float,
    long_offset: float,
    lat_offset: float,
    height_offset: float,
    sensor: str,
) -> Dict[str, Any]:
    data = {
        "pd": pd,
        "pfa": pfa,
        "integration": integration,
        "rcs": rcs,
        "plot": plot,
        "inset_position": inset_position,
        "flip": flip,
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
        "sensor": sensor,
    }
    with open("./settings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return data


@app.callback(
    output={
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
    },
    inputs={"unused_pathname": Input("url", "pathname")},
    prevent_initial_call="initial_duplicate",
)
def load_settings(
    unused_pathname: str,
) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
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
    if os.path.exists("./settings.json"):
        try:
            with open("./settings.json", "r", encoding="utf-8") as f:
                saved = json.load(f)
            defaults.update({k: v for k, v in saved.items() if v is not None and k in defaults})
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


if __name__ == "__main__":
    # app.run_server(debug=True, threaded=True, processes=1, host="0.0.0.0")

    FlaskUI(
        app=server, server="flask", port=34687, profile_dir_prefix="coverage_app"
    ).run()
