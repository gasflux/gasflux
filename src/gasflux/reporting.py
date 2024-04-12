"""This module provides functions for generating mass balance reports."""

from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Template
from plotly.io import to_html

from . import plotting


import json
import numpy as np


def mass_balance_report(
    krig_params: dict,
    wind_fig: go.Figure,
    background_fig: go.Figure,
    threed_fig: go.Figure,
    krig_fig: go.Figure,
    windrose_fig: go.Figure,
) -> str:
    """Generate a mass balance report."""
    template_path = Path(__file__).parents[2] / "templates" / "mass_balance_template.html"

    # Convert the figures to HTML
    plot_htmls = {}
    for name, fig in zip(
        ["3D", "krig", "windrose", "wind", "background"],
        [threed_fig, krig_fig, windrose_fig, wind_fig, background_fig],
        strict=False,
    ):
        if fig:
            plot_htmls[name] = to_html(fig, full_html=False)
        else:
            plot_htmls[name] = plotting.blank_figure()

    summary_data = {
        "Estimated flux": f"{krig_params.get('volume', 0):.3f} kgh⁻¹",
    }

    with Path.open(template_path) as f:
        template_content = f.read()

    template = Template(template_content)
    return template.render(
        title="Mass Balance Report",
        summary_data=summary_data,
        threeD=plot_htmls["3D"],
        krig=plot_htmls["krig"],
        windrose=plot_htmls["windrose"],
        wind=plot_htmls["wind"],
        background=plot_htmls["background"],
    )


def check_and_replace_large_arrays(output_vars: dict, threshold_size: int):
    """
    Iterate through the output_vars dictionary and replace large numpy arrays
    with their metadata (e.g., shape and data type).

    Parameters:
        output_vars (dict): The dictionary containing output data including potential numpy arrays.
        threshold_size (int): The number of elements above which an array is considered large.
    """
    del_keys = []
    for key, value in output_vars.items():
        if isinstance(value, dict):
            output_vars[key] = check_and_replace_large_arrays(value, threshold_size)  # recursive
        elif isinstance(value, np.ndarray):
            if value.size > threshold_size:
                del_keys.append(key)
    for key in del_keys:
        del output_vars[key]
    return output_vars


def save_data(data: dict, filename: str | Path, striplong: bool = True):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError("Unsupported data type")

    if striplong:
        data = check_and_replace_large_arrays(data, threshold_size=50)

    with open(filename, "w") as f:
        json.dump(data, f, default=convert, indent=4)
