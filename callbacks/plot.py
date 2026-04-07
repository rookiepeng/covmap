"""Plot management callbacks: clear, clear-last, hold."""

from typing import Dict, List, Any

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


def register(app):
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
    def clear_plot(unused_clear_btn: Any, unused_plot_type: Any) -> Dict[str, List]:
        """Clear all plots from the display."""
        return {"fig_data": []}

    @app.callback(
        output={
            "fig_data": Output("figure-data", "data", allow_duplicate=True),
        },
        inputs={
            "unused_clear_btn": Input("clear-last-plot", "n_clicks"),
        },
        state={
            "fig_data_input": State("figure-data", "data"),
        },
        prevent_initial_call=True,
    )
    def clear_last_plot(unused_clear_btn: Any, fig_data_input: List) -> Dict[str, List]:
        """Remove the most recently added plot from the display."""
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
    def hold_plot(
        unused_hold_btn: Any, current_figs: List, new_fig: List
    ) -> Dict[str, List]:
        """Add a new plot while maintaining existing plots."""
        return {"fig_data": current_figs + new_fig}
