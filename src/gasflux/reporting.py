"""This module provides functions for generating mass balance reports."""

from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Template
from plotly.io import to_html
from datetime import datetime
import gasflux
import yaml

import logging
from . import plotting


import json
import numpy as np

logger = logging.getLogger(__name__)


def mass_balance_report(
    krig_params: dict,
    wind_fig: go.Figure,
    background_fig: go.Figure,
    threed_fig: go.Figure,
    krig_fig: go.Figure,
    windrose_fig: go.Figure,
) -> str:
    """Generate a mass balance report."""
    template_path = Path(__file__).parent / "templates" / "mass_balance_template.html"

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


def generate_reports(name: str, processor, config: dict):
    """
    Generates reports, configuration files, and processed output variables for gasflux processing runs.

    Parameters:
        name (str): The name identifier for the current processing run.
        processor (object): The processing object containing report data and output variables.
        config (dict): Configuration dictionary used for processing.
    """
    output_dir = Path(config["output_dir"]).expanduser()
    processing_time = datetime.now()
    output_path = output_dir / name / processing_time.strftime("%Y-%m-%d_%H-%M-%S-%f_processing_run")
    output_path.mkdir(parents=True, exist_ok=True)

    # Save reports
    for gas, report in processor.reports.items():
        report_path = output_path / f"{name}_{gas}_report.html"
        with open(report_path, "w", encoding="utf-8") as file:
            file.write(report)

    # Save config with version
    version = gasflux.__version__
    header = f"# Gasflux version: {version}\n# Output config for file {name} from processing run at {processing_time}\n"
    config_path = output_path / f"{name}_config.yaml"
    with open(config_path, "w") as file:
        file.write(header)
        yaml.safe_dump(config, file)

    # Save output variables
    output_vars = processor.output_vars
    output_vars = delete_large_arrays(output_vars, threshold_size=50)
    header = (
        f"# Gasflux version: {version}\n# Output variables for file {name} from processing run at {processing_time}\n"
    )
    filename = output_path / f"{name}_output_vars.json"
    with open(filename, "w") as file:
        file.write(header)
        json.dump(
            output_vars, file, default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item, indent=4
        )
    logger.info(f"Processing run saved to {output_path}")


def delete_large_arrays(output_vars: dict, threshold_size: int) -> dict:
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
            output_vars[key] = delete_large_arrays(value, threshold_size)  # recursive
        elif isinstance(value, np.ndarray):
            if value.size > threshold_size:
                del_keys.append(key)
    for key in del_keys:
        del output_vars[key]
    return output_vars
