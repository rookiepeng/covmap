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
from callbacks.sensor import load_config as _load_config
from callbacks.colors import color_for_idx as _color_for_idx, hex_to_rgba as _hex_to_rgba


def register(app):
    @app.callback(
        output={
            "fig": Output("scatter", "figure"),
            "layers": Output("layers-store", "data", allow_duplicate=True),
            "property_container": Output("property-container", "children"),
        },
        inputs={
            # Only controls and config are Inputs (triggers).
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
            "flip": Input("flip-checklist", "value"),
            "inset_position": Input("inset-position", "value"),
            "long_offset": Input("long", "value"),
            "lat_offset": Input("lat", "value"),
            "height_offset": Input("height", "value"),
            "config": Input("config", "data"),
            "legend_in": Input("legend", "value"),
        },
        state={
            # Layer stores are State — never trigger this callback.
            "min_pd": State("pd", "min"),
            "max_pd": State("pd", "max"),
            "min_pfa": State("pfa", "min"),
            "max_pfa": State("pfa", "max"),
            "active": State("active-layer-store", "data"),
            "layers_in": State("layers-store", "data"),
            "sensor": State("sensor", "value"),
        },
        prevent_initial_call=True,
    )
    def coverage_plot(
        pd, pfa, rcs, fascia_loss, mfg_loss, temp_loss, rain_loss,
        vert_misalign_angle, roll_offset, az_offset, sw_model, plot_type,
        flip, inset_position,
        long_offset, lat_offset, height_offset,
        config,
        min_pd, max_pd, min_pfa, max_pfa,
        active, layers_in, sensor, legend_in,
    ) -> Dict[str, Any]:
        """Recompute the active layer from controls and render all layers."""

        # ── Gate: need config + valid parameter values ──────────────
        if config is None:
            raise PreventUpdate
        if pd is None or pfa is None or rcs is None:
            raise PreventUpdate
        if pd < min_pd or pd > max_pd:
            raise PreventUpdate
        if pfa < min_pfa or pfa > max_pfa:
            raise PreventUpdate
        for v in [fascia_loss, mfg_loss, temp_loss, rain_loss,
                  vert_misalign_angle, roll_offset, az_offset,
                  long_offset, lat_offset, height_offset]:
            if v is None:
                raise PreventUpdate

        # ── Legend: always use what the user has typed ─────────────
        layers = list(layers_in or [])
        active_idx = next((i for i, l in enumerate(layers) if l["id"] == active), None)
        legend = legend_in or ""

        # ── Compute traces for active layer from current controls ───
        fill_enabled = "fill" in (flip or [])
        layer_color = _color_for_idx(active_idx) if active_idx is not None else None
        traces, fig_layout, container = _compute_layer(
            pd, pfa, rcs, fascia_loss, mfg_loss, temp_loss, rain_loss,
            vert_misalign_angle, roll_offset, az_offset, sw_model,
            plot_type, flip, inset_position,
            long_offset, lat_offset, height_offset, config, legend, fill_enabled, layer_color,
        )

        # ── Update the active layer in layers list ──────────────────

        if active_idx is not None:
            layers[active_idx] = {
                **layers[active_idx],
                "traces": traces,
                "name": sensor.replace(".json", "") if sensor else layers[active_idx].get("name", "Layer"),
                "settings": {
                    "sensor": sensor,
                    "pd": pd, "pfa": pfa,
                    "integration": sw_model, "rcs": rcs,
                    "plot": plot_type, "inset_position": inset_position,
                    "flip": flip or [],
                    "fascia": fascia_loss, "mfg": mfg_loss,
                    "temp": temp_loss, "rain": rain_loss,
                    "az_offset": az_offset, "misalign": vert_misalign_angle,
                    "roll_offset": roll_offset,
                    "long_offset": long_offset, "lat_offset": lat_offset,
                    "height_offset": height_offset,
                    "legend": legend,
                },
            }

        # ── Backfill other layers that have settings but no traces ──
        for i, layer in enumerate(layers):
            if i == active_idx or layer.get("traces"):
                continue
            s = layer.get("settings", {})
            other_sensor = s.get("sensor")
            if not other_sensor:
                continue
            try:
                other_config = _load_config("./radar/" + other_sensor)
            except (OSError, KeyError):
                continue
            other_fill = "fill" in (s.get("flip") or [])
            other_color = _color_for_idx(i)
            other_legend = s.get("legend", "")
            try:
                other_traces, _, _ = _compute_layer(
                    s.get("pd", 0.5), s.get("pfa", 0.0001), s.get("rcs", 10),
                    s.get("fascia", 0), s.get("mfg", 0), s.get("temp", 0),
                    s.get("rain", 0), s.get("misalign", 0), s.get("roll_offset", 0),
                    s.get("az_offset", 0), s.get("integration", "Swerling 3"),
                    s.get("plot", "Azimuth Coverage"), s.get("flip", []),
                    s.get("inset_position", "top-left"),
                    s.get("long_offset", 0), s.get("lat_offset", 0),
                    s.get("height_offset", 0), other_config, other_legend,
                    other_fill, other_color,
                )
                layers[i] = {**layer, "traces": other_traces}
            except Exception:
                pass

        # ── Combine all layers' traces; only active gets inset ──────
        all_traces = []
        for i, layer in enumerate(layers):
            if i == active_idx:
                all_traces.extend(traces)
            else:
                # Strip inset traces (heatmap/scatter on secondary axes)
                for t in layer.get("traces", []):
                    if t.get("xaxis") == "x2" or t.get("yaxis") == "y2":
                        continue
                    all_traces.append(t)

        return {
            "fig": go.Figure(data=all_traces, layout=fig_layout).to_dict(),
            "layers": layers,
            "property_container": container,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_layer(
    pd, pfa, rcs, fascia_loss, mfg_loss, temp_loss, rain_loss,
    vert_misalign_angle, roll_offset, az_offset, sw_model,
    plot_type, flip, inset_position,
    long_offset, lat_offset, height_offset, config, legend, fill_enabled=True, layer_color=None,
):
    """Compute traces, layout, and property container for a single layer."""

    flip_az = "flip_az" in (flip or [])
    flip_el = "flip_el" in (flip or [])

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

    # Azimuth pattern
    az_ptn_full = np.array(config["az_ptn"])
    az_ang_full = np.array(config["az_ang"])
    az_ptn_full = az_ptn_full - az_ptn_full[np.where(az_ang_full == chamber_az)]
    if flip_az:
        az_ptn_full = np.flip(az_ptn_full)
        az_ang_full = np.flip(-az_ang_full)

    idx = np.where(np.logical_and(az_ang_full >= az_start - 1, az_ang_full <= az_end + 1))
    az_ang = az_ang_full[idx]
    az_ptn = az_ptn_full[idx].copy()
    az_ptn[0] = -1000
    az_ptn[-1] = -1000

    # Elevation pattern
    el_ptn_full = np.array(config["el_ptn"])
    el_ang_full = np.array(config["el_ang"])
    el_ptn_full = el_ptn_full - el_ptn_full[np.where(el_ang_full == chamber_el)]
    if flip_el:
        el_ptn_full = np.flip(el_ptn_full)
        el_ang_full = np.flip(-el_ang_full)

    idx = np.where(np.logical_and(el_ang_full >= el_start - 1, el_ang_full <= el_end + 1))
    el_ang = el_ang_full[idx]
    el_ptn = el_ptn_full[idx].copy()
    el_ptn[0] = -1000
    el_ptn[-1] = -1000

    min_snr = roc_snr(pfa, pd, 1, sw_model)
    nci_gain = roc_snr(pfa, pd, 1, sw_model) - roc_snr(pfa, pd, channels, sw_model)
    roll_rad = roll_offset / 180 * np.pi

    # Misalignment loss at boresight
    az_at_bs = vert_misalign_angle * np.sin(roll_rad)
    el_at_bs = vert_misalign_angle * np.cos(roll_rad)
    az_loss_bs = np.interp(az_at_bs, az_ang_full, az_ptn_full, left=-1000, right=-1000)
    el_loss_bs = np.interp(el_at_bs, el_ang_full, el_ptn_full, left=-1000, right=-1000)
    el_missalign_loss = az_loss_bs + el_loss_bs

    max_range = 10 ** (
        (chamber_snr + 40 * np.log10(chamber_range) - chamber_rcs + rcs
         + nci_gain + fascia_loss + mfg_loss + temp_loss + rain_loss
         + straddling_loss + extra_gain_loss - min_snr) / 40
    )

    # 2D heatmap for inset
    az_grid, el_grid = np.meshgrid(az_ang, el_ang)
    az_ant_2d = az_grid * np.cos(roll_rad) + el_grid * np.sin(roll_rad)
    el_ant_2d = -az_grid * np.sin(roll_rad) + el_grid * np.cos(roll_rad)
    az_loss_2d = np.interp(az_ant_2d.ravel(), az_ang_full, az_ptn_full, left=-1000, right=-1000).reshape(az_grid.shape)
    el_loss_2d = np.interp(el_ant_2d.ravel(), el_ang_full, el_ptn_full, left=-1000, right=-1000).reshape(el_grid.shape)
    combined_2d = az_loss_2d + el_loss_2d
    oov_2d = (az_ant_2d < az_start) | (az_ant_2d > az_end) | (el_ant_2d < el_start) | (el_ant_2d > el_end)
    combined_2d[oov_2d] = -1000
    range_2d = max_range * 10 ** (combined_2d / 40)
    range_2d[range_2d < 1e-6] = np.nan

    inset_domains = {
        "top-right": {"x": [0.72, 0.98], "y": [0.67, 0.95]},
        "top-left": {"x": [0.04, 0.30], "y": [0.67, 0.95]},
        "bottom-right": {"x": [0.72, 0.98], "y": [0.05, 0.33]},
        "bottom-left": {"x": [0.04, 0.30], "y": [0.05, 0.33]},
    }
    show_inset = inset_position != "hidden" and inset_position in inset_domains
    inset_dom = inset_domains.get(inset_position, {"x": [0.04, 0.30], "y": [0.67, 0.95]})

    inset_heatmap = {
        "type": "heatmap", "z": range_2d.tolist(),
        "x": az_ang.tolist(), "y": el_ang.tolist(),
        "colorscale": "Viridis", "showscale": False,
        "hoverinfo": "x+y+z", "xaxis": "x2", "yaxis": "y2",
    }
    inset_axes = {
        "xaxis2": {"domain": inset_dom["x"], "anchor": "y2", "showgrid": False, "visible": show_inset, "showline": True, "mirror": True, "tickfont": {"size": 9}},
        "yaxis2": {"domain": inset_dom["y"], "anchor": "x2", "showgrid": False, "visible": show_inset, "showline": True, "mirror": True, "tickfont": {"size": 9}},
    }

    if plot_type == "Azimuth Coverage":
        traces, fig_layout = _azimuth_coverage(
            az_ang, az_ang_full, az_ptn_full, el_ang_full, el_ptn_full,
            roll_rad, vert_misalign_angle, az_offset, long_offset, lat_offset,
            az_start, az_end, el_start, el_end, max_range, legend,
            inset_heatmap, show_inset, inset_axes, fill_enabled, layer_color,
        )
    elif plot_type == "Azimuth vs. Range":
        traces, fig_layout = _azimuth_vs_range(
            az_ang, az_ang_full, az_ptn_full, el_ang_full, el_ptn_full,
            roll_rad, vert_misalign_angle,
            az_start, az_end, el_start, el_end, max_range, legend,
            inset_heatmap, show_inset, inset_axes, fill_enabled, layer_color,
        )
    elif plot_type == "Elevation Coverage":
        traces, fig_layout = _elevation_coverage(
            el_ang, az_ang_full, az_ptn_full, el_ang_full, el_ptn_full,
            roll_rad, vert_misalign_angle, az_offset, long_offset, height_offset,
            az_start, az_end, el_start, el_end, max_range, legend,
            inset_heatmap, show_inset, inset_axes, fill_enabled, layer_color,
        )
    elif plot_type == "Elevation vs. Range":
        traces, fig_layout = _elevation_vs_range(
            el_ang, az_ang_full, az_ptn_full, el_ang_full, el_ptn_full,
            roll_rad, az_offset,
            az_start, az_end, el_start, el_end, max_range, legend,
            inset_heatmap, show_inset, inset_axes, fill_enabled, layer_color,
        )

    container = _build_property_container(
        discription, chamber_snr, chamber_rcs, chamber_range,
        channels, min_snr, nci_gain, straddling_loss,
        el_missalign_loss, extra_gain_loss,
    )

    # Convert ndarray values in traces to plain lists for JSON serialization
    serializable_traces = []
    for t in traces:
        st = {}
        for k, v in t.items():
            if isinstance(v, np.ndarray):
                st[k] = v.tolist()
            else:
                st[k] = v
        serializable_traces.append(st)

    return serializable_traces, fig_layout, container


def _make_inset_cut(x_range, y_range):
    return {
        "type": "scatter", "mode": "lines",
        "x": [float(x_range[0]), float(x_range[1])],
        "y": [float(y_range[0]), float(y_range[1])],
        "line": {"color": "red", "width": 2},
        "showlegend": False, "hoverinfo": "skip",
        "xaxis": "x2", "yaxis": "y2",
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


def _apply_color(trace, layer_color, fill_enabled):
    """Attach a stable line color (and semi-transparent fill) to a trace dict."""
    if not layer_color:
        return trace
    trace["line"] = {"color": layer_color}
    if fill_enabled and layer_color.startswith("#"):
        trace["fillcolor"] = _hex_to_rgba(layer_color, 0.2)
    return trace


def _azimuth_coverage(
    az_ang, az_ang_full, az_ptn_full, el_ang_full, el_ptn_full,
    roll_rad, vert_misalign_angle, az_offset, long_offset, lat_offset,
    az_start, az_end, el_start, el_end, max_range, legend,
    inset_heatmap, show_inset, inset_axes, fill_enabled=True, layer_color=None,
):
    az_antenna = az_ang * np.cos(roll_rad) + vert_misalign_angle * np.sin(roll_rad)
    el_antenna = -az_ang * np.sin(roll_rad) + vert_misalign_angle * np.cos(roll_rad)
    az_loss = np.interp(az_antenna, az_ang_full, az_ptn_full, left=-1000, right=-1000)
    el_loss = np.interp(el_antenna, el_ang_full, el_ptn_full, left=-1000, right=-1000)
    combined = az_loss + el_loss
    oov = (az_antenna < az_start) | (az_antenna > az_end) | (el_antenna < el_start) | (el_antenna > el_end)
    combined[oov] = -1000
    coverage = max_range * 10 ** (combined / 40)
    x = coverage * np.cos((az_ang + az_offset) / 180 * np.pi) + long_offset
    y = coverage * np.sin((az_ang + az_offset) / 180 * np.pi) + lat_offset
    trace = {"mode": "lines", "type": "scatter", "x": x, "y": y, "fill": "tozeroy" if fill_enabled else "none", "name": legend}
    traces = [_apply_color(trace, layer_color, fill_enabled)]
    if show_inset:
        traces.extend([inset_heatmap, _make_inset_cut([az_ang[0], az_ang[-1]], [vert_misalign_angle, vert_misalign_angle])])
    return traces, _base_layout("Longitude (m)", "Latitude (m)", inset_axes, scale_y_to_x=True)


def _azimuth_vs_range(
    az_ang, az_ang_full, az_ptn_full, el_ang_full, el_ptn_full,
    roll_rad, vert_misalign_angle,
    az_start, az_end, el_start, el_end, max_range, legend,
    inset_heatmap, show_inset, inset_axes, fill_enabled=True, layer_color=None,
):
    az_antenna = az_ang * np.cos(roll_rad) + vert_misalign_angle * np.sin(roll_rad)
    el_antenna = -az_ang * np.sin(roll_rad) + vert_misalign_angle * np.cos(roll_rad)
    az_loss = np.interp(az_antenna, az_ang_full, az_ptn_full, left=-1000, right=-1000)
    el_loss = np.interp(el_antenna, el_ang_full, el_ptn_full, left=-1000, right=-1000)
    combined = az_loss + el_loss
    oov = (az_antenna < az_start) | (az_antenna > az_end) | (el_antenna < el_start) | (el_antenna > el_end)
    combined[oov] = -1000
    coverage = max_range * 10 ** (combined / 40)
    trace = {"mode": "lines", "type": "scatter", "x": az_ang, "y": coverage, "fill": "tozeroy" if fill_enabled else "none", "name": legend}
    traces = [_apply_color(trace, layer_color, fill_enabled)]
    if show_inset:
        traces.extend([inset_heatmap, _make_inset_cut([az_ang[0], az_ang[-1]], [vert_misalign_angle, vert_misalign_angle])])
    return traces, _base_layout("Azimuth (deg)", "Range (m)", inset_axes)


def _elevation_coverage(
    el_ang, az_ang_full, az_ptn_full, el_ang_full, el_ptn_full,
    roll_rad, vert_misalign_angle, az_offset, long_offset, height_offset,
    az_start, az_end, el_start, el_end, max_range, legend,
    inset_heatmap, show_inset, inset_axes, fill_enabled=True, layer_color=None,
):
    az_ant = az_offset * np.cos(roll_rad) + el_ang * np.sin(roll_rad)
    el_ant = -az_offset * np.sin(roll_rad) + el_ang * np.cos(roll_rad)
    az_loss = np.interp(az_ant, az_ang_full, az_ptn_full, left=-1000, right=-1000)
    el_loss = np.interp(el_ant, el_ang_full, el_ptn_full, left=-1000, right=-1000)
    combined = az_loss + el_loss
    oov = (az_ant < az_start) | (az_ant > az_end) | (el_ant < el_start) | (el_ant > el_end)
    combined[oov] = -1000
    coverage = max_range * 10 ** (combined / 40)
    x = coverage * np.cos((el_ang + vert_misalign_angle) / 180 * np.pi) + long_offset
    y = coverage * np.sin((el_ang + vert_misalign_angle) / 180 * np.pi) + height_offset
    trace = {"mode": "lines", "type": "scatter", "x": x, "y": y, "fill": "tozeroy" if fill_enabled else "none", "name": legend}
    traces = [_apply_color(trace, layer_color, fill_enabled)]
    if show_inset:
        traces.extend([inset_heatmap, _make_inset_cut([az_offset, az_offset], [el_ang[0], el_ang[-1]])])
    return traces, _base_layout("Longitude (m)", "Height (m)", inset_axes, scale_y_to_x=True)


def _elevation_vs_range(
    el_ang, az_ang_full, az_ptn_full, el_ang_full, el_ptn_full,
    roll_rad, az_offset,
    az_start, az_end, el_start, el_end, max_range, legend,
    inset_heatmap, show_inset, inset_axes, fill_enabled=True, layer_color=None,
):
    az_ant = az_offset * np.cos(roll_rad) + el_ang * np.sin(roll_rad)
    el_ant = -az_offset * np.sin(roll_rad) + el_ang * np.cos(roll_rad)
    az_loss = np.interp(az_ant, az_ang_full, az_ptn_full, left=-1000, right=-1000)
    el_loss = np.interp(el_ant, el_ang_full, el_ptn_full, left=-1000, right=-1000)
    combined = az_loss + el_loss
    oov = (az_ant < az_start) | (az_ant > az_end) | (el_ant < el_start) | (el_ant > el_end)
    combined[oov] = -1000
    coverage = max_range * 10 ** (combined / 40)
    trace = {"mode": "lines", "type": "scatter", "x": el_ang, "y": coverage, "fill": "tozeroy" if fill_enabled else "none", "name": legend}
    traces = [_apply_color(trace, layer_color, fill_enabled)]
    if show_inset:
        traces.extend([inset_heatmap, _make_inset_cut([az_offset, az_offset], [el_ang[0], el_ang[-1]])])
    return traces, _base_layout("Elevation (deg)", "Range (m)", inset_axes)


def _build_property_container(
    discription, chamber_snr, chamber_rcs, chamber_range,
    channels, min_snr, nci_gain, straddling_loss,
    el_missalign_loss, extra_gain_loss,
):
    container = []
    if discription is not None:
        container.append(dbc.FormText(dcc.Markdown(discription, dangerously_allow_html=True), color="primary"))
    container.append(dbc.FormText(f"Chamber SNR: {chamber_snr} dB ({chamber_rcs} dBsm at {chamber_range} m)"))
    container.append(dbc.FormText(f"Number of NCI Channels: {channels}"))
    container.append(dbc.FormText(f"Min SNR: {round(min_snr, 3)} dB"))
    container.append(dbc.FormText(f"Integration Gain: {round(nci_gain, 3)} dB"))
    container.append(dbc.FormText(f"Straddling Loss: {round(straddling_loss, 3)} dB"))
    container.append(dbc.FormText(f"Misalignment Loss: {round(el_missalign_loss, 3)} dB"))
    container.append(dbc.FormText(f"Extra Processing Gain/Loss: {round(extra_gain_loss, 3)} dB"))
    return container
