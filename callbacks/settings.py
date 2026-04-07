"""Settings persistence callbacks: save and load."""

from typing import Dict, List, Any

import json
import os

from dash.dependencies import Input, Output


def register(app):
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
    def load_settings(unused_pathname: str) -> Dict[str, Any]:
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
                defaults.update(
                    {k: v for k, v in saved.items() if v is not None and k in defaults}
                )
            except (json.JSONDecodeError, OSError):
                pass
        return defaults
