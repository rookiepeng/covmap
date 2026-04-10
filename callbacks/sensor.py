"""Sensor management callbacks: reload, upload, and select."""

from typing import Dict, Any, Union, Optional

import json
import os
import base64

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


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
    """
    with open(json_file, "r", encoding="utf-8") as read_file:
        return json.load(read_file)


def register(app):
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
        """Reload the list of available radar sensor configurations."""

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
        """Process and save an uploaded radar configuration file."""

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
        """Configure display parameters based on selected radar sensor."""

        if sensor is None:
            raise PreventUpdate

        config = load_config("./radar/" + sensor)

        if "requirement" in config:
            # Requirement configs define angle vs range directly;
            # they have no elevation FOV, so disable misalignment.
            return {
                "config": config,
                "misalign_min": 0,
                "misalign_max": 0,
                "misalign": 0,
                "sensor_store": sensor,
            }

        misalign_min = config["el_fov"][0]
        misalign_max = config["el_fov"][1]

        if misalgin_state is not None and misalign_min <= misalgin_state <= misalign_max:
            misalign = dash.no_update
        else:
            misalign = 0
        return {
            "config": config,
            "misalign_min": misalign_min,
            "misalign_max": misalign_max,
            "misalign": misalign,
            "sensor_store": sensor,
        }
