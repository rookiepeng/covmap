"""Settings persistence callbacks: save and load layers state."""

from typing import Dict, List, Any

import json
import os

from dash.dependencies import Input, Output


_LAYER1_ID = "layer-1"

_DEFAULT_LAYER_SETTINGS = {
    "sensor": None,
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
    "legend": "",
}


def _default_layer1():
    return {
        "id": _LAYER1_ID,
        "name": "Layer 1",
        "settings": dict(_DEFAULT_LAYER_SETTINGS),
        "traces": [],
    }


def register(app):
    @app.callback(
        Output("app-settings", "data"),
        Input("layers-store", "data"),
        Input("active-layer-store", "data"),
        prevent_initial_call=True,
    )
    def save_layers(layers, active):
        data = {"layers": layers or [], "active": active}
        try:
            with open("./settings.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except OSError:
            pass
        return data

    @app.callback(
        output={
            "layers": Output("layers-store", "data", allow_duplicate=True),
            "active": Output("active-layer-store", "data", allow_duplicate=True),
        },
        inputs={"unused_pathname": Input("url", "pathname")},
        prevent_initial_call="initial_duplicate",
    )
    def load_layers(unused_pathname: str) -> Dict[str, Any]:
        if os.path.exists("./settings.json"):
            try:
                with open("./settings.json", "r", encoding="utf-8") as f:
                    saved = json.load(f)

                # New format: has "layers" key
                if "layers" in saved and saved["layers"]:
                    return {
                        "layers": saved["layers"],
                        "active": saved.get("active", saved["layers"][0]["id"]),
                    }

                # Old format: flat key/value settings — migrate to Layer 1
                settings = dict(_DEFAULT_LAYER_SETTINGS)
                old_key_map = {
                    "sensor": "sensor", "pd": "pd", "pfa": "pfa",
                    "integration": "integration", "rcs": "rcs",
                    "plot": "plot", "inset_position": "inset_position",
                    "flip": "flip", "fascia": "fascia", "mfg": "mfg",
                    "temp": "temp", "rain": "rain",
                    "az_offset": "az_offset", "misalign": "misalign",
                    "roll_offset": "roll_offset", "long_offset": "long_offset",
                    "lat_offset": "lat_offset", "height_offset": "height_offset",
                }
                for old_k, new_k in old_key_map.items():
                    if old_k in saved and saved[old_k] is not None:
                        settings[new_k] = saved[old_k]
                layer1 = _default_layer1()
                layer1["settings"] = settings
                return {"layers": [layer1], "active": _LAYER1_ID}

            except (json.JSONDecodeError, OSError):
                pass

        # No settings file — create a fresh Layer 1
        return {"layers": [_default_layer1()], "active": _LAYER1_ID}

