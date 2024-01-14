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
    Load config json file

    :param str json_file
        json file path

    :return: configuration struct
    :rtype: dict
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
    Callback when reload button clicked
    Only run once when the page is loaded

    Scan the json files under `radar` folder

    :param btn button click count
    :param sensor_store local stored selection
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
    """_summary_

    :param list_of_contents: _description_
    :type list_of_contents: _type_
    :param list_of_names: _description_
    :type list_of_names: _type_
    :param unused_list_of_dates: _description_
    :type unused_list_of_dates: _type_
    :param n_clicks: _description_
    :type n_clicks: _type_
    :raises PreventUpdate: _description_
    :return: _description_
    :rtype: _type_
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
    Callback when a sensor is selected

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
    Callback when `Clear plots` button is clicked

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
    Callback when `Clear plots` button is clicked

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
    Callback when `Hold plot` button is clicked

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
    Plot the coverage map

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
    Export the current figure as the .png file

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
    Export the current figure as the .svg file

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
    Export the current figure as the .html file

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
    Export the raw data as the .csv file

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
    """_summary_

    :param rcs_input: _description_
    :type rcs_input: _type_
    :param rcs_slider: _description_
    :type rcs_slider: _type_
    :raises PreventUpdate: _description_
    :return: _description_
    :rtype: _type_
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
    """_summary_

    :param fascia_input: _description_
    :type fascia_input: _type_
    :param fascia_slider: _description_
    :type fascia_slider: _type_
    :raises PreventUpdate: _description_
    :return: _description_
    :rtype: _type_
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
    """_summary_

    :param mfg_input: _description_
    :type mfg_input: _type_
    :param mfg_slider: _description_
    :type mfg_slider: _type_
    :raises PreventUpdate: _description_
    :return: _description_
    :rtype: _type_
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
    """_summary_

    :param temp_input: _description_
    :type temp_input: _type_
    :param temp_slider: _description_
    :type temp_slider: _type_
    :raises PreventUpdate: _description_
    :return: _description_
    :rtype: _type_
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
    """_summary_

    :param rain_input: _description_
    :type rain_input: _type_
    :param rain_slider: _description_
    :type rain_slider: _type_
    :raises PreventUpdate: _description_
    :return: _description_
    :rtype: _type_
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
    """_summary_

    :param misalign_input: _description_
    :type misalign_input: _type_
    :param misalign_slider: _description_
    :type misalign_slider: _type_
    :raises PreventUpdate: _description_
    :return: _description_
    :rtype: _type_
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
    """_summary_

    :param azoffset_input: _description_
    :type azoffset_input: _type_
    :param az_slider: _description_
    :type az_slider: _type_
    :raises PreventUpdate: _description_
    :return: _description_
    :rtype: _type_
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
    """_summary_

    :param long_input: _description_
    :type long_input: _type_
    :param long_slider: _description_
    :type long_slider: _type_
    :raises PreventUpdate: _description_
    :return: _description_
    :rtype: _type_
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
    """_summary_

    :param lat_input: _description_
    :type lat_input: _type_
    :param lat_slider: _description_
    :type lat_slider: _type_
    :raises PreventUpdate: _description_
    :return: _description_
    :rtype: _type_
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
    """_summary_

    :param height_input: _description_
    :type height_input: _type_
    :param height_slider: _description_
    :type height_slider: _type_
    :raises PreventUpdate: _description_
    :return: _description_
    :rtype: _type_
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
