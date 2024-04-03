"""This module provides functions for generating mass balance reports."""
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Template
from plotly.io import to_html

from . import plotting


def mass_balance_report(krig_params: dict,
                        wind_fig: go.Figure,
                        baseline_fig: go.Figure,
                        threed_fig: go.Figure,
                        krig_fig: go.Figure,
                        windrose_fig: go.Figure) -> str:
    """Generate a mass balance report."""
    template_path = Path(__file__).parents[2] / "templates" / "mass_balance_template.html"

    # Convert the figures to HTML
    plot_htmls = {}
    for name, fig in zip(["3D", "krig", "windrose", "wind", "baseline"],
                         [threed_fig, krig_fig, windrose_fig, wind_fig, baseline_fig], strict=False):
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
        baseline=plot_htmls["baseline"],
    )
