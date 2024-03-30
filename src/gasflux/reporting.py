from pathlib import Path

from jinja2 import Template
from plotly.io import to_html

from . import plotting


def mass_balance_report(df, krig_params, wind_fig, baseline_fig, threed_fig, krig_fig, windrose_fig):
    template_path = Path(__file__).parents[2] / "templates" / "mass_balance_template.html"

    try:
        dfk = krig_params
    except KeyError as e:
        print(f"Missing krig params: {e}")
        dfk = {}

    # Convert the figures to HTML
    plot_htmls = {}
    for name, fig in zip(["3D", "krig", "windrose", "wind", "baseline"],
                         [threed_fig, krig_fig, windrose_fig, wind_fig, baseline_fig]):
        if fig:
            plot_htmls[name] = to_html(fig, full_html=False)
        else:
            plot_htmls[name] = plotting.blank_figure()

    summary_data = {
        "Estimated flux": f"{dfk.get('volume', 0):.3f} kgh⁻¹",
    }

    with open(template_path) as f:
        template_content = f.read()

    template = Template(template_content)
    rendered_report = template.render(
        title="Mass Balance Report",
        summary_data=summary_data,
        threeD=plot_htmls["3D"],
        krig=plot_htmls["krig"],
        windrose=plot_htmls["windrose"],
        wind=plot_htmls["wind"],
        baseline=plot_htmls["baseline"],
    )

    return rendered_report
