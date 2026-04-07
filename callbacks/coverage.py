"""Coverage plot callback — the main computation and rendering."""

from typing import Dict, List, Any

import dash
from dash import dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go

from roc.tools import roc_snr


def register(app):
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
        """Generate a coverage plot based on the input parameters."""

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
        flip_az = "flip_az" in flip
        flip_el = "flip_el" in flip

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

        inset_axes = {
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

        if plot_type == "Azimuth Coverage":
            new_fig, fig_layout = _azimuth_coverage(
                az_ang,
                az_ang_full,
                az_ptn_full,
                el_ang_full,
                el_ptn_full,
                roll_rad,
                vert_misalign_angle,
                az_offset,
                long_offset,
                lat_offset,
                az_start,
                az_end,
                el_start,
                el_end,
                max_range,
                legend,
                inset_heatmap,
                show_inset,
                inset_axes,
            )
        elif plot_type == "Azimuth vs. Range":
            new_fig, fig_layout = _azimuth_vs_range(
                az_ang,
                az_ang_full,
                az_ptn_full,
                el_ang_full,
                el_ptn_full,
                roll_rad,
                vert_misalign_angle,
                az_start,
                az_end,
                el_start,
                el_end,
                max_range,
                legend,
                inset_heatmap,
                show_inset,
                inset_axes,
            )
        elif plot_type == "Elevation Coverage":
            new_fig, fig_layout = _elevation_coverage(
                el_ang,
                az_ang_full,
                az_ptn_full,
                el_ang_full,
                el_ptn_full,
                roll_rad,
                vert_misalign_angle,
                az_offset,
                long_offset,
                height_offset,
                az_start,
                az_end,
                el_start,
                el_end,
                max_range,
                legend,
                inset_heatmap,
                show_inset,
                inset_axes,
            )
        elif plot_type == "Elevation vs. Range":
            new_fig, fig_layout = _elevation_vs_range(
                el_ang,
                az_ang_full,
                az_ptn_full,
                el_ang_full,
                el_ptn_full,
                roll_rad,
                az_offset,
                az_start,
                az_end,
                el_start,
                el_end,
                max_range,
                legend,
                inset_heatmap,
                show_inset,
                inset_axes,
            )

        container = _build_property_container(
            discription,
            chamber_snr,
            chamber_rcs,
            chamber_range,
            channels,
            min_snr,
            nci_gain,
            straddling_loss,
            el_missalign_loss,
            extra_gain_loss,
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


# ---------------------------------------------------------------------------
# Helper functions (not callbacks, just pure computation)
# ---------------------------------------------------------------------------


def _make_inset_cut(x_range, y_range):
    return {
        "type": "scatter",
        "mode": "lines",
        "x": [float(x_range[0]), float(x_range[1])],
        "y": [float(y_range[0]), float(y_range[1])],
        "line": {"color": "red", "width": 2},
        "showlegend": False,
        "hoverinfo": "skip",
        "xaxis": "x2",
        "yaxis": "y2",
    }


def _base_layout(xaxis_title, yaxis_title, inset_axes, scale_y_to_x=False):
    layout = {
        "template": "seaborn",
        "margin": {"l": 60, "r": 10, "t": 30, "b": 50},
        "xaxis": {"title": {"text": xaxis_title}, "domain": [0, 1]},
        "yaxis": {"title": {"text": yaxis_title}},
    }
    if scale_y_to_x:
        layout["yaxis"]["scaleanchor"] = "x"
        layout["yaxis"]["scaleratio"] = 1
    layout.update(inset_axes)
    return layout


def _azimuth_coverage(
    az_ang,
    az_ang_full,
    az_ptn_full,
    el_ang_full,
    el_ptn_full,
    roll_rad,
    vert_misalign_angle,
    az_offset,
    long_offset,
    lat_offset,
    az_start,
    az_end,
    el_start,
    el_end,
    max_range,
    legend,
    inset_heatmap,
    show_inset,
    inset_axes,
):
    az_antenna = az_ang * np.cos(roll_rad) + vert_misalign_angle * np.sin(roll_rad)
    el_antenna = -az_ang * np.sin(roll_rad) + vert_misalign_angle * np.cos(roll_rad)
    az_loss = np.interp(az_antenna, az_ang_full, az_ptn_full, left=-1000, right=-1000)
    el_loss = np.interp(el_antenna, el_ang_full, el_ptn_full, left=-1000, right=-1000)
    combined_ptn = az_loss + el_loss
    out_of_fov = (
        (az_antenna < az_start)
        | (az_antenna > az_end)
        | (el_antenna < el_start)
        | (el_antenna > el_end)
    )
    combined_ptn[out_of_fov] = -1000
    coverage = max_range * 10 ** (combined_ptn / 40)
    coverage_long = coverage * np.cos((az_ang + az_offset) / 180 * np.pi) + long_offset
    coverage_lat = coverage * np.sin((az_ang + az_offset) / 180 * np.pi) + lat_offset

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
    if show_inset:
        inset_cut = _make_inset_cut(
            [az_ang[0], az_ang[-1]], [vert_misalign_angle, vert_misalign_angle]
        )
        new_fig.extend([inset_heatmap, inset_cut])

    fig_layout = _base_layout(
        "Longitude (m)", "Latitude (m)", inset_axes, scale_y_to_x=True
    )
    return new_fig, fig_layout


def _azimuth_vs_range(
    az_ang,
    az_ang_full,
    az_ptn_full,
    el_ang_full,
    el_ptn_full,
    roll_rad,
    vert_misalign_angle,
    az_start,
    az_end,
    el_start,
    el_end,
    max_range,
    legend,
    inset_heatmap,
    show_inset,
    inset_axes,
):
    az_antenna = az_ang * np.cos(roll_rad) + vert_misalign_angle * np.sin(roll_rad)
    el_antenna = -az_ang * np.sin(roll_rad) + vert_misalign_angle * np.cos(roll_rad)
    az_loss = np.interp(az_antenna, az_ang_full, az_ptn_full, left=-1000, right=-1000)
    el_loss = np.interp(el_antenna, el_ang_full, el_ptn_full, left=-1000, right=-1000)
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
    if show_inset:
        inset_cut = _make_inset_cut(
            [az_ang[0], az_ang[-1]], [vert_misalign_angle, vert_misalign_angle]
        )
        new_fig.extend([inset_heatmap, inset_cut])

    fig_layout = _base_layout("Azimuth (deg)", "Range (m)", inset_axes)
    return new_fig, fig_layout


def _elevation_coverage(
    el_ang,
    az_ang_full,
    az_ptn_full,
    el_ang_full,
    el_ptn_full,
    roll_rad,
    vert_misalign_angle,
    az_offset,
    long_offset,
    height_offset,
    az_start,
    az_end,
    el_start,
    el_end,
    max_range,
    legend,
    inset_heatmap,
    show_inset,
    inset_axes,
):
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
        coverage * np.cos((el_ang + vert_misalign_angle) / 180 * np.pi) + long_offset
    )
    coverage_height = (
        coverage * np.sin((el_ang + vert_misalign_angle) / 180 * np.pi) + height_offset
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
    if show_inset:
        inset_cut = _make_inset_cut([az_offset, az_offset], [el_ang[0], el_ang[-1]])
        new_fig.extend([inset_heatmap, inset_cut])

    fig_layout = _base_layout(
        "Longitude (m)", "Height (m)", inset_axes, scale_y_to_x=True
    )
    return new_fig, fig_layout


def _elevation_vs_range(
    el_ang,
    az_ang_full,
    az_ptn_full,
    el_ang_full,
    el_ptn_full,
    roll_rad,
    az_offset,
    az_start,
    az_end,
    el_start,
    el_end,
    max_range,
    legend,
    inset_heatmap,
    show_inset,
    inset_axes,
):
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
    if show_inset:
        inset_cut = _make_inset_cut([az_offset, az_offset], [el_ang[0], el_ang[-1]])
        new_fig.extend([inset_heatmap, inset_cut])

    fig_layout = _base_layout("Elevation (deg)", "Range (m)", inset_axes)
    return new_fig, fig_layout


def _build_property_container(
    discription,
    chamber_snr,
    chamber_rcs,
    chamber_range,
    channels,
    min_snr,
    nci_gain,
    straddling_loss,
    el_missalign_loss,
    extra_gain_loss,
):
    container = []
    if discription is not None:
        container.append(
            dbc.FormText(
                dcc.Markdown(discription, dangerously_allow_html=True), color="primary"
            )
        )
    container.append(
        dbc.FormText(
            f"Chamber SNR: {chamber_snr} dB ({chamber_rcs} dBsm at {chamber_range} m)"
        )
    )
    container.append(dbc.FormText(f"Number of NCI Channels: {channels}"))
    container.append(dbc.FormText(f"Min SNR: {round(min_snr, 3)} dB"))
    container.append(dbc.FormText(f"Integration Gain: {round(nci_gain, 3)} dB"))
    container.append(dbc.FormText(f"Straddling Loss: {round(straddling_loss, 3)} dB"))
    container.append(
        dbc.FormText(f"Misalignment Loss: {round(el_missalign_loss, 3)} dB")
    )
    container.append(
        dbc.FormText(f"Extra Processing Gain/Loss: {round(extra_gain_loss, 3)} dB")
    )
    return container
