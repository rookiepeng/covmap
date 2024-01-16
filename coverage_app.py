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

    :param json_file: The path to the JSON file.
    :type json_file: str
    :return: The loaded configuration as a dictionary.
    :rtype: dict
    :example:
    >>> config = load_config("config.json")"""

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

    :param unused_btn: Unused parameter, can be any value.
    :param sensor_store: The current sensor value.
    :type sensor_store: str
    :return: A dictionary containing updated sensor options and values.
    :rtype: dict
    :example:
    >>> result = reload("unused", "sensor1.json")
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

    :param list_of_contents: The base64-encoded contents of the file.
    :type list_of_contents: str
    :param list_of_names: The name of the file.
    :type list_of_names: str
    :param unused_list_of_dates: Unused parameter, can be any value.
    :param n_clicks: The number of times the upload button has been clicked.
    :type n_clicks: int
    :return: A dictionary containing the updated "reload" value and the "sensor_store" value.
    :rtype: dict
    :raises PreventUpdate: If the list_of_contents is None.
    :example:
    >>> result = upload_config("base64-encoded-contents", "config.json", "unused", 3)
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

    :param sensor: The name of the sensor configuration file.
    :type sensor: str
    :param misalgin_state: The current misalignment state.
    :type misalgin_state: float
    :return: A dictionary containing the sensor configuration,
        misalignment range and state, and other relevant information.
    :rtype: dict
    :example:
    >>> result = sensor_select("sensor1.json", 0.5)
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

    :param unused_clear_btn: Unused parameter, can be any value.
    :param unused_plot_type: Unused parameter, can be any value.
    :return: A dictionary containing an empty list for "fig_data".
    :rtype: dict
    :example:
    >>> result = clear_plot("unused", "unused")
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

    :param unused_clear_btn: Unused parameter, can be any value.
    :param unused_plot_type: Unused parameter, can be any value.
    :param fig_data_input: The input list of figure data.
    :type fig_data_input: list
    :return: A dictionary containing the updated "fig_data" list after removing the last plot.
    :rtype: dict
    :raises PreventUpdate: If the fig_data_input list is empty.
    :example:
    >>> result = clear_last_plot("unused", "unused", [{"x": [1, 2, 3], "y": [4, 5, 6]}])
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
def hold_plot(unused_hold_btn, current_figs, new_fig):
    """
    Hold and add a new plot.

    :param unused_hold_btn: Unused parameter, can be any value.
    :param current_figs: The current list of figure data.
    :type current_figs: list
    :param new_fig: The new figure data to be added.
    :type new_fig: list
    :return: A dictionary containing the updated "fig_data" list after adding the new plot.
    :rtype: dict
    :example:
    >>> result = hold_plot("unused", [{"x": [1, 2, 3], "y": [4, 5, 6]}],
    ...     [{"x": [7, 8, 9], "y": [10, 11, 12]}])
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
    Generate a coverage plot based on the input parameters.

    :param pd: Probability of detection.
    :type pd: float
    :param pfa: Probability of false alarm.
    :type pfa: float
    :param rcs: Radar cross section.
    :type rcs: float
    :param fascia_loss: Fascia loss.
    :type fascia_loss: float
    :param mfg_loss: MFG loss.
    :type mfg_loss: float
    :param temp_loss: Temperature loss.
    :type temp_loss: float
    :param rain_loss: Rain loss.
    :type rain_loss: float
    :param vert_misalign_angle: Vertical misalignment angle.
    :type vert_misalign_angle: float
    :param az_offset: Azimuth offset.
    :type az_offset: float
    :param sw_model: Software model.
    :type sw_model: str
    :param plot_type: Type of coverage plot.
    :type plot_type: str
    :param new_legend_entry: New legend entry.
    :type new_legend_entry: str
    :param min_pd: Minimum probability of detection.
    :type min_pd: float
    :param max_pd: Maximum probability of detection.
    :type max_pd: float
    :param min_pfa: Minimum probability of false alarm.
    :type min_pfa: float
    :param max_pfa: Maximum probability of false alarm.
    :type max_pfa: float
    :param fig_data: Existing figure data.
    :type fig_data: list
    :param flip: Flip patterns.
    :type flip: list
    :param long_offset: Longitude offset.
    :type long_offset: float
    :param lat_offset: Latitude offset.
    :type lat_offset: float
    :param height_offset: Height offset.
    :type height_offset: float
    :param config: Configuration parameters.
    :type config: dict
    :return: A dictionary containing the updated figure data,
        figure layout, property container, and legend entry.
    :rtype: dict
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

    :param unused_n_clicks: Unused parameter, can be any value.
    :param fig: The figure object to be exported.
    :return: Sends the exported PNG file to the user for download.
    :rtype: dcc.send_file
    :example:
    >>> result = export_png("unused", fig)
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

    :param unused_n_clicks: Unused parameter, can be any value.
    :param fig: The figure object to be exported.
    :return: Sends the exported SVG file to the user for download.
    :rtype: dcc.send_file
    :example:
    >>> result = export_svg("unused", fig)
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

    :param unused_n_clicks: Unused parameter, can be any value.
    :param fig: The figure object to be exported.
    :return: Sends the exported HTML file to the user for download.
    :rtype: dcc.send_file
    :example:
    >>> result = export_html("unused", fig)
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

    :param unused_n_clicks: Unused parameter, can be any value.
    :param data: The data to be exported.
    :param plot_type: The type of plot associated with the data.
    :return: Sends the exported CSV file to the user for download.
    :rtype: dcc.send_data_frame
    :example:
    >>> result = export_data("unused", data, "Azimuth Coverage")
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

    :param rcs_input: The input value of RCS.
    :param rcs_slider: The value of RCS selected using a slider.
    :return: A tuple containing the linked values for RCS.
    :rtype: tuple
    :raises: PreventUpdate if the RCS input value is None.
    :example:
    >>> result = link_rcs(10, None)
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

    :param fascia_input: The input value of Fascia.
    :param fascia_slider: The value of Fascia selected using a slider.
    :return: A tuple containing the linked values for Fascia.
    :rtype: tuple
    :raises: PreventUpdate if the Fascia input value is None.
    :example:
    >>> result = link_fascia(1, None)
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

    :param mfg_input: The input value of Manufacturing Loss.
    :param mfg_slider: The value of Manufacturing Loss selected using a slider.
    :return: A tuple containing the linked values for Manufacturing Loss.
    :rtype: tuple
    :raises: PreventUpdate if the Manufacturing Loss input value is None.
    :example:
    >>> result = link_mfg(2, None)
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

    :param temp_input: The input value of Temperature Loss.
    :param temp_slider: The value of Temperature Loss selected using a slider.
    :return: A tuple containing the linked values for Temperature Loss.
    :rtype: tuple
    :raises: PreventUpdate if the Temperature Loss input value is None.
    :example:
    >>> result = link_temp(-3, None)
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

    :param rain_input: The input value of Rain Loss.
    :param rain_slider: The value of Rain Loss selected using a slider.
    :return: A tuple containing the linked values for Rain Loss.
    :rtype: tuple
    :raises: PreventUpdate if the Rain Loss input value is None.
    :example:
    >>> result = link_rain(-3, None)
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

    :param misalign_input: The input value of Vertical Misalignment Angle.
    :param misalign_slider: The value of Vertical Misalignment Angle selected using a slider.
    :return: A tuple containing the linked values for Vertical Misalignment Angle.
    :rtype: tuple
    :raises: PreventUpdate if the Vertical Misalignment Angle input value is None.
    :example:
    >>> result = link_misalign(0, None)
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

    :param azoffset_input: The input value of Azimuth Offset.
    :param az_slider: The value of Azimuth Offset selected using a slider.
    :return: A tuple containing the linked values for Azimuth Offset.
    :rtype: tuple
    :raises: PreventUpdate if the Azimuth Offset input value is None.
    :example:
    >>> result = link_azoffset(0, None)
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

    :param long_input: The input value of Longitude Offset.
    :param long_slider: The value of Longitude Offset selected using a slider.
    :return: A tuple containing the linked values for Longitude Offset.
    :rtype: tuple
    :raises: PreventUpdate if the Longitude Offset input value is None.
    :example:
    >>> result = link_long(0, None)
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

    :param lat_input: The input value of Latitude Offset.
    :param lat_slider: The value of Latitude Offset selected using a slider.
    :return: A tuple containing the linked values for Latitude Offset.
    :rtype: tuple
    :raises: PreventUpdate if the Latitude Offset input value is None.
    :example:
    >>> result = link_lat(0, None)
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

    :param height_input: The input value of Height Offset.
    :param height_slider: The value of Height Offset selected using a slider.
    :return: A tuple containing the linked values for Height Offset.
    :rtype: tuple
    :raises: PreventUpdate if the Height Offset input value is None.
    :example:
    >>> result = link_height(0, None)
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
