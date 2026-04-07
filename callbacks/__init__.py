from callbacks.sensor import register as register_sensor
from callbacks.coverage import register as register_coverage
from callbacks.layers import register as register_layers
from callbacks.export import register as register_export
from callbacks.settings import register as register_settings


def register_all(app):
    register_sensor(app)
    register_layers(app)
    register_coverage(app)
    register_export(app)
    register_settings(app)
