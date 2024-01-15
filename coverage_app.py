"""
    Copyright (C) 2023 - PRESENT  Zhengyu Peng
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


def load_config(json_file):
    """
    Load configuration from a JSON file.

    Parameters:
    - json_file (str): The path to the JSON file.

    Returns:
    - dict: The loaded configuration as a dictionary.

    Example Usage:
    config = load_config("config.json")
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
def reload(unused_btn, sensor_store):
    """
    Reload sensor options and values.

    Parameters:
    - unused_btn: Unused parameter, can be any value.
    - sensor_store (str): The current sensor value.

    Returns:
    - dict: A dictionary containing updated sensor options and values.

    Example Usage:
    result = reload("unused", "sensor1.json")
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
def upload_config(list_of_contents, list_of_names, unused_list_of_dates, n_clicks):
    """
    Upload and save a configuration file.

    Parameters:
    - list_of_contents (str): The base64-encoded contents of the file.
    - list_of_names (str): The name of the file.
    - unused_list_of_dates: Unused parameter, can be any value.
    - n_clicks (int): The number of times the upload button has been clicked.

    Returns:
    - dict: A dictionary containing the updated "reload" value and the "sensor_store" value.

    Raises:
    - PreventUpdate: If the list_of_contents is None.

    Example Usage:
    result = upload_config("base64-encoded-contents", "config.json", "unused", 3)
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
def sensor_select(sensor, misalgin_state):
    """
    Select and configure a sensor.

    Parameters:
    - sensor (str): The name of the sensor configuration file.
    - misalgin_state (float): The current misalignment state.

    Returns:
    - dict: A dictionary containing the sensor configuration,
        misalignment range and state, and other relevant information.

    Example Usage:
    result = sensor_select("sensor1.json", 0.5)
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
def clear_plot(unused_clear_btn, unused_plot_type):
    """
    Clear the plot.

    Parameters:
    - unused_clear_btn: Unused parameter, can be any value.
    - unused_plot_type: Unused parameter, can be any value.

    Returns:
    - dict: A dictionary containing an empty list for "fig_data".

    Example Usage:
    result = clear_plot("unused", "unused")
    """

    return {"fig_data": []}


@app.callback(
    output={
        "fig_data": Output("figure-data", "data", allow_duplicate=True),
    },
    inputs={
        "unused_plot_type": Input("plot", "value"),
        "unused_clear_btn": Input("clear-last-plot", "n_clicks"),
    },
    state={
        "fig_data_input": State("figure-data", "data"),
    },
    prevent_initial_call=True,
)
def clear_last_plot(unused_clear_btn, unused_plot_type, fig_data_input):
    """
    Clear the last plot.

    Parameters:
    - unused_clear_btn: Unused parameter, can be any value.
    - unused_plot_type: Unused parameter, can be any value.
    - fig_data_input (list): The input list of figure data.

    Returns:
    - dict: A dictionary containing the updated "fig_data" list after removing the last plot.

    Raises:
    - PreventUpdate: If the fig_data_input list is empty.

    Example Usage:
    result = clear_last_plot("unused", "unused", [{"x": [1, 2, 3], "y": [4, 5, 6]}])
    """

    if len(fig_data_input) > 0:
        fig_data_input.pop(-1)
        return {"fig_data": fig_data_input}
    else:
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
def hold_plot(unused_hold_btn, current_figs, new_fig):
    """
    Hold and add a new plot.

    Parameters:
    - unused_hold_btn: Unused parameter, can be any value.
    - current_figs (list): The current list of figure data.
    - new_fig (list): The new figure data to be added.

    Returns:
    - dict: A dictionary containing the updated "fig_data" list after adding the new plot.

    Example Usage:
    result = hold_plot("unused", [{"x": [1, 2, 3], "y": [4, 5, 6]}],
        [{"x": [7, 8, 9], "y": [10, 11, 12]}])
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
    pd,
    pfa,
    rcs,
    fascia_loss,
    mfg_loss,
    temp_loss,
    rain_loss,
    vert_misalign_angle,
    az_offset,
    sw_model,
    plot_type,
    new_legend_entry,
    min_pd,
    max_pd,
    min_pfa,
    max_pfa,
    fig_data,
    flip,
    long_offset,
    lat_offset,
    height_offset,
    config,
):
    """
    Plot the coverage map.

    Parameters:
    - pd (float): Probability of Detection.
    - pfa (float): Probability of False Alarm.
    - rcs (float): Radar Cross Section.
    - fascia_loss (float): Fascia Loss.
    - mfg_loss (float): Manufacturing Loss.
    - temp_loss (float): Temperature Loss.
    - rain_loss (float): Rain Loss.
    - vert_misalign_angle (float): Vertical Misalignment Angle.
    - az_offset (float): Azimuth Offset.
    - sw_model (str): Software Model.
    - plot_type (str): Type of plot to generate.
    - new_legend_entry (str): New legend entry for the plot.
    - min_pd (float): Minimum Probability of Detection.
    - max_pd (float): Maximum Probability of Detection.
    - min_pfa (float): Minimum Probability of False Alarm.
    - max_pfa (float): Maximum Probability of False Alarm.
    - fig_data (list): Existing figure data.
    - flip (list): List of flip options.
    - long_offset (float): Longitude offset.
    - lat_offset (float): Latitude offset.
    - height_offset (float): Height offset.
    - config (dict): Sensor configuration.

    Returns:
    - dict: A dictionary containing the updated figure, new figure data,
        property container, and legend entry.

    Raises:
    - PreventUpdate: If any of the required inputs are None or outside the specified range.

    Example Usage:
    result = coverage_plot(
        0.9,
        1e-5,
        -30,
        1,
        -1,
        -2,
        -3,
        0,
        0,
        "Model A",
        "Azimuth Coverage",
        "New Entry",
        0.8,
        0.95,
        1e-6,
        1e-4,
        [],
        ["flip_az"],
        0,
        0,
        0,
        config
    )
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
def export_png(unused_n_clicks, fig):
    """
    Export the current figure as a PNG file.

    Parameters:
    - unused_n_clicks: Unused parameter, can be any value.
    - fig: The figure object to be exported.

    Returns:
    - dcc.send_file: Sends the exported PNG file to the user for download.

    Example Usage:
    result = export_png("unused", fig)
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
def export_svg(unused_n_clicks, fig):
    """
    Export the current figure as an SVG file.

    Parameters:
    - unused_n_clicks: Unused parameter, can be any value.
    - fig: The figure object to be exported.

    Returns:
    - dcc.send_file: Sends the exported SVG file to the user for download.

    Example Usage:
    result = export_svg("unused", fig)
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
def export_html(unused_n_clicks, fig):
    """
    Export the current figure as an HTML file.

    Parameters:
    - unused_n_clicks: Unused parameter, can be any value.
    - fig: The figure object to be exported.

    Returns:
    - dcc.send_file: Sends the exported HTML file to the user for download.

    Example Usage:
    result = export_html("unused", fig)
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
def export_data(unused_n_clicks, data, plot_type):
    """
    Export the raw data as a CSV file.

    Parameters:
    - unused_n_clicks: Unused parameter, can be any value.
    - data: The data to be exported.
    - plot_type: The type of plot associated with the data.

    Returns:
    - dcc.send_data_frame: Sends the exported CSV file to the user for download.

    Example Usage:
    result = export_data("unused", data, "Azimuth Coverage")
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
def link_rcs(rcs_input, rcs_slider):
    """
    Link the RCS input and slider values.

    Parameters:
    - rcs_input: The input value of RCS.
    - rcs_slider: The value of RCS selected using a slider.

    Returns:
    - tuple: A tuple containing the linked values for RCS.

    Raises:
    - PreventUpdate: If the RCS input value is None.

    Example Usage:
    result = link_rcs(10, None)
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
def link_fascia(fascia_input, fascia_slider):
    """
    Link the Fascia input and slider values.

    Parameters:
    - fascia_input: The input value of Fascia.
    - fascia_slider: The value of Fascia selected using a slider.

    Returns:
    - tuple: A tuple containing the linked values for Fascia.

    Raises:
    - PreventUpdate: If the Fascia input value is None.

    Example Usage:
    result = link_fascia(1, None)
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
def link_mfg(mfg_input, mfg_slider):
    """
    Link the Manufacturing Loss input and slider values.

    Parameters:
    - mfg_input: The input value of Manufacturing Loss.
    - mfg_slider: The value of Manufacturing Loss selected using a slider.

    Returns:
    - tuple: A tuple containing the linked values for Manufacturing Loss.

    Raises:
    - PreventUpdate: If the Manufacturing Loss input value is None.

    Example Usage:
    result = link_mfg(2, None)
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
def link_temp(temp_input, temp_slider):
    """
    Link the Temperature Loss input and slider values.

    Parameters:
    - temp_input: The input value of Temperature Loss.
    - temp_slider: The value of Temperature Loss selected using a slider.

    Returns:
    - tuple: A tuple containing the linked values for Temperature Loss.

    Raises:
    - PreventUpdate: If the Temperature Loss input value is None.

    Example Usage:
    result = link_temp(-3, None)
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
def link_rain(rain_input, rain_slider):
    """
    Link the Rain Loss input and slider values.

    Parameters:
    - rain_input: The input value of Rain Loss.
    - rain_slider: The value of Rain Loss selected using a slider.

    Returns:
    - tuple: A tuple containing the linked values for Rain Loss.

    Raises:
    - PreventUpdate: If the Rain Loss input value is None.

    Example Usage:
    result = link_rain(-3, None)
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
def link_misalign(misalign_input, misalign_slider):
    """
    Link the Vertical Misalignment Angle input and slider values.

    Parameters:
    - misalign_input: The input value of Vertical Misalignment Angle.
    - misalign_slider: The value of Vertical Misalignment Angle selected using a slider.

    Returns:
    - tuple: A tuple containing the linked values for Vertical Misalignment Angle.

    Raises:
    - PreventUpdate: If the Vertical Misalignment Angle input value is None.

    Example Usage:
    result = link_misalign(0, None)
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
def link_azoffset(azoffset_input, az_slider):
    """
    Link the Azimuth Offset input and slider values.

    Parameters:
    - azoffset_input: The input value of Azimuth Offset.
    - az_slider: The value of Azimuth Offset selected using a slider.

    Returns:
    - tuple: A tuple containing the linked values for Azimuth Offset.

    Raises:
    - PreventUpdate: If the Azimuth Offset input value is None.

    Example Usage:
    result = link_azoffset(0, None)
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
def link_long(long_input, long_slider):
    """
    Link the Longitude Offset input and slider values.

    Parameters:
    - long_input: The input value of Longitude Offset.
    - long_slider: The value of Longitude Offset selected using a slider.

    Returns:
    - tuple: A tuple containing the linked values for Longitude Offset.

    Raises:
    - PreventUpdate: If the Longitude Offset input value is None.

    Example Usage:
    result = link_long(0, None)
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
def link_lat(lat_input, lat_slider):
    """
    Link the Latitude Offset input and slider values.

    Parameters:
    - lat_input: The input value of Latitude Offset.
    - lat_slider: The value of Latitude Offset selected using a slider.

    Returns:
    - tuple: A tuple containing the linked values for Latitude Offset.

    Raises:
    - PreventUpdate: If the Latitude Offset input value is None.

    Example Usage:
    result = link_lat(0, None)
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
def link_height(height_input, height_slider):
    """
    Link the Height Offset input and slider values.

    Parameters:
    - height_input: The input value of Height Offset.
    - height_slider: The value of Height Offset selected using a slider.

    Returns:
    - tuple: A tuple containing the linked values for Height Offset.

    Raises:
    - PreventUpdate: If the Height Offset input value is None.

    Example Usage:
    result = link_height(0, None)
    """

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "height-input" and height_input is None:
        raise PreventUpdate

    value = height_input if trigger_id == "height-input" else height_slider
    return value, value


if __name__ == "__main__":
    # app.run_server(debug=True, threaded=True, processes=1, host="0.0.0.0")
    FlaskUI(app=server, server="flask").run()
