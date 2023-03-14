import os
from pathlib import Path

from jinja2 import Template

from . import plotting


def mass_balance(dfs, threed_figs, krig_figs, windrose_figs, outlier_figs, baseline_figs, krig_params):
    template_path = Path(__file__).parents[1] / "templates" / "mass_balance_template.html"
    for name, df in dfs.items():
        output_path = Path(os.getcwd()) / "reports" / name
        output_path.mkdir(parents=True, exist_ok=True)
        try:
            dfk = krig_params[name]
        except KeyError as e:
            print(f"Missing krig params in {name}: {e}")
            continue
        plots = {}
        try:
            plots["3D"] = threed_figs[name]
        except KeyError as e:
            print(f"Missing 3D plot in {name}: {e}")
            plots["3D"] = plotting.blank_figure()
        try:
            plots["krig"] = krig_figs[name][0]
        except KeyError as e:
            print(f"Missing krig plot in {name}: {e}")
            plots["krig"] = plotting.blank_figure()
        try:
            plots["windrose"] = windrose_figs[name]
        except KeyError as e:
            print(f"Missing windrose plot in {name}: {e}")
            plots["windrose"] = plotting.blank_figure()
        try:
            plots["outliers"] = outlier_figs[name]
        except KeyError as e:
            print(f"Missing outliers plot in {name}: {e}")
            plots["outliers"] = plotting.blank_figure()
        try:
            plots["baseline"] = baseline_figs[name]
        except KeyError as e:
            print(f"Missing baseline plot in {name}: {e}")
            plots["baseline"] = plotting.blank_figure()

        plot_paths = {}
        for plottitle, plot in plots.items():
            plot_path = output_path / f"{name}_{plottitle}.html"
            plot.write_html(plot_path)
            plot_paths[plottitle] = plot_path

        summary_data = {
            "Estimated flux": f"{dfk['volume']:.3f} kgh⁻¹",
        }

        with open(template_path) as f:
            template_content = f.read()
        template = Template(template_content)
        rendered_template = template.render(
            summary_data=summary_data,
            threeD=plot_paths["3D"],
            krig=plot_paths["krig"],
            windrose=plot_paths["windrose"],
            outliers=plot_paths["outliers"],
            baseline=plot_paths["baseline"],
        )
        with open(output_path / f"{name}_mass_balance_report.html", "w") as f:
            f.write(rendered_template)
