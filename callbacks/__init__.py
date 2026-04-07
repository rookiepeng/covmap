from callbacks.sensor import register as register_sensor
from callbacks.coverage import register as register_coverage
from callbacks.plot import register as register_plot
from callbacks.export import register as register_export
from callbacks.settings import register as register_settings


def register_all(app):
    register_sensor(app)
    register_coverage(app)
    register_plot(app)
    register_export(app)
    register_settings(app)
