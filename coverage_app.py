"""
    Copyright (C) 2023 - PRESENT  Zhengyu Peng
    
    Coverage Map Application - A tool for visualizing radar coverage patterns.
"""

import json
import os
import base64

import dash
from dash import dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

import numpy as np
import plotly.io as pio
import plotly.graph_objects as go

import pandas as pds

from flaskwebgui import FlaskUI

from roc.tools import roc_snr

from layout.layout import get_app_layout

from typing import Dict, List, Any, Tuple, Union, Optional

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
    n_clicks: int
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
        "misalign": Output("misalign", "value"),
        "misalign_input_min": Output("misalign-input", "min"),
        "misalign_input_max": Output("misalign-input", "max"),
        "misalign_input": Output("misalign-input", "value"),
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
        "misalign_input_min": misalign_min,
        "misalign_input_max": misalign_max,
        "misalign_input": misalign,
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
    unused_hold_btn: Any,
    current_figs: List,
    new_fig: List
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
        "rcs": Input("rcs-input", "value"),
        "fascia_loss": Input("fascia-input", "value"),
        "mfg_loss": Input("mfg-input", "value"),
        "temp_loss": Input("temp-input", "value"),
        "rain_loss": Input("rain-input", "value"),
        "vert_misalign_angle": Input("misalign-input", "value"),
        "az_offset": Input("az-offset-input", "value"),
        "sw_model": Input("integration", "value"),
        "plot_type": Input("plot", "value"),
        "fig_data": Input("figure-data", "data"),
        "new_legend_entry": Input("legend", "value"),
        "flip": Input("flip-checklist", "value"),
        "long_offset": Input("long-input", "value"),
        "lat_offset": Input("lat-input", "value"),
        "height_offset": Input("height-input", "value"),
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
    long_offset: float,
    lat_offset: float,
    height_offset: float,
    config: Dict[str, Any]
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
    az_ptn = np.array(config["az_ptn"])
    az_ang = np.array(config["az_ang"])
    az_ptn = az_ptn - az_ptn[np.where(az_ang == chamber_az)]
    if flip_az:
        az_ptn = np.flip(az_ptn)
        az_ang = np.flip(-az_ang)

    idx = np.where(np.logical_and(az_ang >= az_start - 1, az_ang <= az_end + 1))
    az_ang = az_ang[idx]
    az_ptn = az_ptn[idx]
    az_ptn[0] = -1000
    az_ptn[-1] = -1000

    # normalize elevation pattern based on chamber angle
    el_ptn = np.array(config["el_ptn"])
    el_ang = np.array(config["el_ang"])
    el_ptn = el_ptn - el_ptn[np.where(el_ang == chamber_el)]
    if flip_el:
        el_ptn = np.flip(el_ptn)
        el_ang = np.flip(-el_ang)

    idx = np.where(np.logical_and(el_ang >= el_start - 1, el_ang <= el_end + 1))
    el_ang = el_ang[idx]
    el_ptn = el_ptn[idx]
    el_ptn[0] = -1000
    el_ptn[-1] = -1000

    min_snr = roc_snr(pfa, pd, 1, sw_model)
    nci_gain = roc_snr(pfa, pd, 1, sw_model) - roc_snr(pfa, pd, channels, sw_model)

    el_missalign_loss = el_ptn[np.where(el_ang == vert_misalign_angle)[0][0]]
    el_missalign_loss_linear = 10 ** (el_missalign_loss / 40)

    if az_offset >= az_start and az_offset <= az_end:
        az_offset_loss = az_ptn[np.where(az_ang == az_offset)[0][0]]
    else:
        az_offset_loss = -100000
    az_offset_loss_linear = 10 ** (az_offset_loss / 40)

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
            + ", "
            + str(abs(fascia_loss))
            + " dB Fascia Loss, "
            + str(abs(temp_loss))
            + " dB Temp Loss,<br>"
            + str(abs(rain_loss))
            + " dB Rain Loss, "
            + str(abs(mfg_loss))
            + " dB MFG, "
            + str(vert_misalign_angle)
            + " deg Misalignment"
        )

    # clear all the held plots if the plot type is changed
    if trigger_id == "plot":
        fig_data = []

    if plot_type == "Azimuth Coverage":
        coverage = max_range * 10 ** (az_ptn / 40) * el_missalign_loss_linear
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
        fig_layout = {
            "template": pio.templates["seaborn"],
            "margin": {"l": 20, "r": 5, "t": 30, "b": 20},
            "xaxis": {"title": "Longitude (m)"},
            "yaxis": {"title": "Latitude (m)", "scaleanchor": "x", "scaleratio": 1},
        }
    elif plot_type == "Azimuth vs. Range":
        coverage = max_range * 10 ** (az_ptn / 40) * el_missalign_loss_linear
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
        fig_layout = {
            "template": pio.templates["seaborn"],
            "margin": {"l": 20, "r": 5, "t": 30, "b": 20},
            "xaxis": {"title": "Azimuth (deg)"},
            "yaxis": {"title": "Range (m)"},
        }
    elif plot_type == "Elevation Coverage":
        coverage = max_range * 10 ** (el_ptn / 40) * az_offset_loss_linear
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
        fig_layout = {
            "template": pio.templates["seaborn"],
            "margin": {"l": 20, "r": 5, "t": 30, "b": 20},
            "xaxis": {"title": "Longitude (m)"},
            "yaxis": {"title": "Height (m)", "scaleanchor": "x", "scaleratio": 1},
        }
    elif plot_type == "Elevation vs. Range":
        coverage = max_range * 10 ** (el_ptn / 40) * az_offset_loss_linear
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
        fig_layout = {
            "template": pio.templates["seaborn"],
            "margin": {"l": 20, "r": 5, "t": 30, "b": 20},
            "xaxis": {"title": "Elevation (deg)"},
            "yaxis": {"title": "Range (m)"},
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
        "fig": {
            "data": fig_data + new_fig,
            "layout": fig_layout,
        },
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
    unused_n_clicks: Any,
    data: List[Dict[str, List[float]]],
    plot_type: str
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
    Output("rcs-input", "value"),
    Output("rcs", "value"),
    Input("rcs-input", "value"),
    Input("rcs", "value"),
)
def link_rcs(
    rcs_input: Optional[float],
    rcs_slider: float
) -> Tuple[float, float]:
    """
    Link the RCS input and slider values.

    Parameters
    ----------
    rcs_input : float
        The input value of RCS
    rcs_slider : float
        The value of RCS selected using a slider

    Returns
    -------
    tuple
        A tuple containing the linked values for RCS

    Raises
    ------
    PreventUpdate
        If the RCS input value is None
    """

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "rcs-input" and rcs_input is None:
        raise PreventUpdate

    value = rcs_input if trigger_id == "rcs-input" else rcs_slider
    return value, value


@app.callback(
    Output("fascia-input", "value"),
    Output("fascia", "value"),
    Input("fascia-input", "value"),
    Input("fascia", "value"),
)
def link_fascia(
    fascia_input: Optional[float],
    fascia_slider: float
) -> Tuple[float, float]:
    """
    Link the Fascia input and slider values.

    Parameters
    ----------
    fascia_input : float
        The input value of Fascia
    fascia_slider : float
        The value of Fascia selected using a slider

    Returns
    -------
    tuple
        A tuple containing the linked values for Fascia

    Raises
    ------
    PreventUpdate
        If the Fascia input value is None
    """

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "fascia-input" and fascia_input is None:
        raise PreventUpdate

    value = fascia_input if trigger_id == "fascia-input" else fascia_slider
    return value, value


@app.callback(
    Output("mfg-input", "value"),
    Output("mfg", "value"),
    Input("mfg-input", "value"),
    Input("mfg", "value"),
)
def link_mfg(
    mfg_input: Optional[float],
    mfg_slider: float
) -> Tuple[float, float]:
    """
    Link the Manufacturing Loss input and slider values.

    Parameters
    ----------
    mfg_input : float
        The input value of Manufacturing Loss
    mfg_slider : float
        The value of Manufacturing Loss selected using a slider

    Returns
    -------
    tuple
        A tuple containing the linked values for Manufacturing Loss

    Raises
    ------
    PreventUpdate
        If the Manufacturing Loss input value is None
    """

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "mfg-input" and mfg_input is None:
        raise PreventUpdate

    value = mfg_input if trigger_id == "mfg-input" else mfg_slider
    return value, value


@app.callback(
    Output("temp-input", "value"),
    Output("temp", "value"),
    Input("temp-input", "value"),
    Input("temp", "value"),
)
def link_temp(
    temp_input: Optional[float],
    temp_slider: float
) -> Tuple[float, float]:
    """
    Link the Temperature Loss input and slider values.

    Parameters
    ----------
    temp_input : float
        The input value of Temperature Loss
    temp_slider : float
        The value of Temperature Loss selected using a slider

    Returns
    -------
    tuple
        A tuple containing the linked values for Temperature Loss

    Raises
    ------
    PreventUpdate
        If the Temperature Loss input value is None
    """

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "temp-input" and temp_input is None:
        raise PreventUpdate

    value = temp_input if trigger_id == "temp-input" else temp_slider
    return value, value


@app.callback(
    Output("rain-input", "value"),
    Output("rain", "value"),
    Input("rain-input", "value"),
    Input("rain", "value"),
)
def link_rain(
    rain_input: Optional[float],
    rain_slider: float
) -> Tuple[float, float]:
    """
    Link the Rain Loss input and slider values.

    Parameters
    ----------
    rain_input : float
        The input value of Rain Loss
    rain_slider : float
        The value of Rain Loss selected using a slider

    Returns
    -------
    tuple
        A tuple containing the linked values for Rain Loss

    Raises
    ------
    PreventUpdate
        If the Rain Loss input value is None
    """

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "rain-input" and rain_input is None:
        raise PreventUpdate

    value = rain_input if trigger_id == "rain-input" else rain_slider
    return value, value


@app.callback(
    Output("misalign-input", "value", allow_duplicate=True),
    Output("misalign", "value", allow_duplicate=True),
    Input("misalign-input", "value"),
    Input("misalign", "value"),
    prevent_initial_call=True,
)
def link_misalign(
    misalign_input: Optional[float],
    misalign_slider: float
) -> Tuple[float, float]:
    """
    Link the Vertical Misalignment Angle input and slider values.

    Parameters
    ----------
    misalign_input : float
        The input value of Vertical Misalignment Angle
    misalign_slider : float
        The value of Vertical Misalignment Angle selected using a slider

    Returns
    -------
    tuple
        A tuple containing the linked values for Vertical Misalignment Angle

    Raises
    ------
    PreventUpdate
        If the Vertical Misalignment Angle input value is None
    """

    ctx = dash.callback_context
    tri_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if tri_id == "misalign-input" and misalign_input is None:
        raise PreventUpdate

    value = misalign_input if tri_id == "misalign-input" else misalign_slider
    return value, value


@app.callback(
    Output("az-offset-input", "value"),
    Output("az-offset", "value"),
    Input("az-offset-input", "value"),
    Input("az-offset", "value"),
)
def link_azoffset(
    azoffset_input: Optional[float],
    az_slider: float
) -> Tuple[float, float]:
    """
    Link the Azimuth Offset input and slider values.

    Parameters
    ----------
    azoffset_input : float
        The input value of Azimuth Offset
    az_slider : float
        The value of Azimuth Offset selected using a slider

    Returns
    -------
    tuple
        A tuple containing the linked values for Azimuth Offset

    Raises
    ------
    PreventUpdate
        If the Azimuth Offset input value is None
    """

    ctx = dash.callback_context
    tri_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if tri_id == "az-offset-input" and azoffset_input is None:
        raise PreventUpdate

    value = azoffset_input if tri_id == "az-offset-input" else az_slider
    return value, value


@app.callback(
    Output("long-input", "value"),
    Output("long", "value"),
    Input("long-input", "value"),
    Input("long", "value"),
)
def link_long(
    long_input: Optional[float],
    long_slider: float
) -> Tuple[float, float]:
    """
    Link the Longitude Offset input and slider values.

    Parameters
    ----------
    long_input : float
        The input value of Longitude Offset
    long_slider : float
        The value of Longitude Offset selected using a slider

    Returns
    -------
    tuple
        A tuple containing the linked values for Longitude Offset

    Raises
    ------
    PreventUpdate
        If the Longitude Offset input value is None
    """

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "long-input" and long_input is None:
        raise PreventUpdate

    value = long_input if trigger_id == "long-input" else long_slider
    return value, value


@app.callback(
    Output("lat-input", "value"),
    Output("lat", "value"),
    Input("lat-input", "value"),
    Input("lat", "value"),
)
def link_lat(
    lat_input: Optional[float],
    lat_slider: float
) -> Tuple[float, float]:
    """
    Link the Latitude Offset input and slider values.

    Parameters
    ----------
    lat_input : float
        The input value of Latitude Offset
    lat_slider : float
        The value of Latitude Offset selected using a slider

    Returns
    -------
    tuple
        A tuple containing the linked values for Latitude Offset

    Raises
    ------
    PreventUpdate
        If the Latitude Offset input value is None
    """

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "lat-input" and lat_input is None:
        raise PreventUpdate

    value = lat_input if trigger_id == "lat-input" else lat_slider
    return value, value


@app.callback(
    Output("height-input", "value"),
    Output("height", "value"),
    Input("height-input", "value"),
    Input("height", "value"),
)
def link_height(
    height_input: Optional[float],
    height_slider: float
) -> Tuple[float, float]:
    """
    Link the Height Offset input and slider values.

    Parameters
    ----------
    height_input : float
        The input value of Height Offset
    height_slider : float
        The value of Height Offset selected using a slider

    Returns
    -------
    tuple
        A tuple containing the linked values for Height Offset

    Raises
    ------
    PreventUpdate
        If the Height Offset input value is None
    """

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "height-input" and height_input is None:
        raise PreventUpdate

    value = height_input if trigger_id == "height-input" else height_slider
    return value, value


if __name__ == "__main__":
    # app.run_server(debug=True, threaded=True, processes=1, host="0.0.0.0")

    FlaskUI(app=server, server="flask", port=34687, profile_dir_prefix="coverage_app").run()
