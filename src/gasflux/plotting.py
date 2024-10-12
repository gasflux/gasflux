"""Various plotting functions mainly based around plotly."""

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import simplekml
from plotly.subplots import make_subplots

from . import processing

pio.templates["default"] = go.layout.Template(
    layout=go.Layout(
        margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0),
    ),
)

pio.templates.default = "simple_white+default"


styling = {
    "colorscale": "geyser",
}


def blank_figure():
    fig = go.Figure()
    return fig


def scatter_3d(
    df: pd.DataFrame,
    color: str = "",
    colorbar_title: str = "",
    timestamp: str = "timestamp",
    x: str = "utm_easting",
    y: str = "utm_northing",
    z: str = "height_ato",
    courses: bool = False,
):
    fig = px.scatter_3d(df, x=x, y=y, z=z)

    if color:
        custom_data = [df[timestamp]]
        if courses:
            custom_data.extend([df["course_elevation"], df["course_azimuth"]])
        custom_data = np.stack(custom_data, axis=-1)
        hover_template = [
            f"{x}: %{{x:.2f}}",
            f"{y}: %{{y:.2f}}",
            f"{z}: %{{z:.2f}}",
            f"{color}: %{{marker.color:.2f}}",
            f"{timestamp}: %{{customdata[0]|%Y-%m-%d %H:%M:%S}}",
            "Index: %{pointNumber}",
        ]

        if courses:
            hover_template.extend(
                [
                    "Course Elevation: %{customdata[1]:.2f}",
                    "Course Azimuth: %{customdata[2]:.2f}",
                ]
            )
        hover_template_str = "<br>".join(hover_template)
        fig.update_traces(
            marker=dict(
                color=df[color],
                size=4,
                opacity=0.5,
                colorscale=styling["colorscale"],
                colorbar=dict(title=colorbar_title),
            ),
            customdata=custom_data,
            hovertemplate=hover_template_str,
        )

    return fig


def scatter_2d(
    df: pd.DataFrame,
    x: str,
    color: str,
    y: str = "height_ato",
    **kwargs,
):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        color_continuous_scale=styling["colorscale"],
        opacity=0.8,
        **kwargs,
    )
    fig.update_traces(
        customdata=df.index,
        hovertemplate="<br>".join(
            [
                "x: %{x:.2f}",
                "height_ato: %{y:.2f}",
                f"{color}: %{{marker.color:.2f}}",
                "Time: %{customdata}",
            ],
        ),
    )

    return fig


def time_series(
    df: pd.DataFrame,
    ys: str | list[str],
    x: str = "timestamp",
    color: str | None = None,
    split=None,
    y_mins: float | list[float | int] | None = None,
    rolling_average: bool = True,
    scatter: bool = True,
    rolling_window: int = 5,
    y_titles: str | list[str] | None = None,
    legend: bool = True,
) -> go.Figure:
    colors = px.colors.qualitative.Plotly

    if isinstance(ys, str):
        ys = [ys]
    if y_titles is None:
        y_titles = ys
        single_title = False
    elif isinstance(y_titles, str):
        y_titles = [y_titles]
        single_title = True
    elif isinstance(y_titles, list):
        if len(y_titles) != len(ys):
            raise ValueError("Length of y_titles must be equal to length of ys")
        single_title = False
    else:
        raise ValueError("Invalid y_titles value")
    if isinstance(y_mins, (float | int)):
        y_mins = [y_mins]
    if isinstance(y_mins, list):
        if len(y_mins) != len(ys):
            raise ValueError("Length of y_mins must be equal to length of ys")

    fig = go.Figure()

    axis_space = 0.05
    domain_start = axis_space * (len(ys)) if len(ys) > 1 else 0
    fig.update_layout(
        xaxis=dict(
            domain=[domain_start, 1],
        ),
    )

    for i, y in enumerate(ys):
        yaxis_name = f"yaxis{i+1}"
        yaxis_ref = f"y{i+1}"

        trace_color = "black" if single_title and i == 0 else colors[i % len(colors)]

        marker_i = dict(size=8, opacity=0.3 if rolling_average else 0.5, color=trace_color)
        if color is not None:
            marker_i["color"] = df[color]  # type: ignore
            marker_i["colorscale"] = styling["colorscale"]

        hover_template = f"{x}: %{{x}}<br>{y}: %{{y:.2f}}<br>"
        if color:
            hover_template += f"{color}: %{{marker.color:.2f}}<br>"

        if scatter:
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[y],
                    name=y,
                    mode="markers",
                    marker=marker_i,
                    yaxis=yaxis_ref,
                    hovertemplate=hover_template,
                    showlegend=legend,
                )
            )

        if rolling_average:
            df[f"rolling_avg_{i}"] = df[y].rolling(window=rolling_window, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[f"rolling_avg_{i}"],
                    name=f"{y} {rolling_window}-point avg",
                    mode="lines",
                    line=dict(color=trace_color, width=2),
                    yaxis=yaxis_ref,
                    showlegend=legend,
                )
            )

        y_data = df[y]
        y_min_var = y_data.min()
        y_max_var = y_data.max()
        y_range = y_max_var - y_min_var or y_max_var * 0.05

        y_axis_min = y_mins[i] if y_mins is not None and y_mins[i] is not None else y_min_var - y_range * 0.05
        y_axis_max = y_max_var + y_range * 0.05

        if single_title and i == 0:
            axis_title = y_titles[0]
            axis_titlefont = dict(color="black")
        elif not single_title:
            axis_title = y_titles[i]
            axis_titlefont = dict(color=trace_color)
        else:
            axis_title = None
            axis_titlefont = {}

        axis_config = dict(
            title=axis_title,
            titlefont=axis_titlefont,
            tickfont=dict(color=trace_color),
            range=[y_axis_min, y_axis_max],
            side="left",
            position=axis_space * i if i > 0 else None,
            anchor="free" if i > 0 else None,
            overlaying="y" if i > 0 else None,
            showgrid=(i == 0),
        )

        fig.layout[yaxis_name] = axis_config

    if split is not None:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=split,
            y0=0,
            x1=split,
            y1=1,
            line=dict(color="red", width=2),
        )

    return fig


def background_plotting(df: pd.DataFrame, gas: str):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ymin = df[gas].min()
    ymax = df[gas].max()
    ylim = [ymin * 0.95, ymax * 1.05]
    y2min = df[f"{gas}_normalised"].min()
    y2lim = (y2min, y2min + (ylim[1] - ylim[0]))
    fig.update_yaxes(range=ylim, secondary_y=False, title_text=f"Sensor {gas} (ppm)")
    fig.update_yaxes(range=y2lim, secondary_y=True, title_text=f"Normalised {gas} (ppm)")
    fig.add_scatter(x=df["timestamp"], y=df[gas], opacity=0.3, name="Raw Data")
    fig.add_scatter(
        x=df["timestamp"], y=df[f"{gas}_fit"], mode="lines", name="Fitted Background", line=dict(dash="dash")
    )
    fig.add_scatter(
        x=df["timestamp"], y=df[f"{gas}_normalised"], yaxis="y2", name="Normalised Data", mode="lines", opacity=0.5
    )
    fig.add_scatter(
        x=df["timestamp"],
        y=np.where(df[f"{gas}_signal"], df[f"{gas}_normalised"], np.nan),
        yaxis="y2",
        name="Classed as signal",
        mode="lines",
        opacity=0.5,
        # color
        # mode="markers",
        # marker=dict(size=3),
    )

    return fig


def windrose_process(df: pd.DataFrame):
    beaufort = {
        "0": [0, 1],
        "1": [1, 2],
        "2": [2, 4],
        "3": [4, 6],
        "4": [6, 9],
        "5": [9, 11],
        "6": [11, 14],
        "7": [14, 17],
        "8": [17, 21],
        "9": [21, 25],
        "10": [25, 29],
        "11": [29, 33],
        "12": [33, 200],
    }

    beaufort_ms = {
        "0": "0-1",
        "1": "1-2",
        "2": "2-4",
        "3": "4-6",
        "4": "6-9",
        "5": "9-11",
        "6": "11-14",
        "7": "14-17",
        "8": "17-21",
        "9": "21-25",
        "10": "25-29",
        "11": "29-33",
        "12": "33+",
    }

    cardinals = {
        "N1": [0, 11.25],
        "NNE": [11.25, 33.75],
        "NE": [33.75, 56.25],
        "ENE": [56.25, 78.75],
        "E": [78.75, 101.25],
        "ESE": [101.25, 123.75],
        "SE": [123.75, 146.25],
        "SSE": [146.25, 168.75],
        "S": [168.75, 191.25],
        "SSW": [191.25, 213.75],
        "SW": [213.75, 236.25],
        "WSW": [236.25, 258.75],
        "W": [258.75, 281.25],
        "WNW": [281.25, 303.75],
        "NW": [303.75, 326.25],
        "NNW": [326.25, 348.75],
        "N2": [348.75, 360],
    }
    df["wind_direction_bin"] = pd.cut(
        df["winddir"],
        bins=[lower for lower, upper in cardinals.values()] + [list(cardinals.values())[-1][1]],
        labels=[key for key in cardinals],
        right=False,
    )

    df["wind_direction_bin"] = (
        df["wind_direction_bin"].map(lambda x: "N" if x in ["N1", "N2"] else x).astype("category")
    )
    df["beaufort"] = pd.cut(
        df["windspeed"],
        bins=[lower for lower, upper in beaufort.values()] + [list(beaufort.values())[-1][1]],
        labels=[key for key in beaufort],
        right=False,
    )
    df["beaufort_ms"] = df["beaufort"].map(beaufort_ms)
    df_windrose = df.groupby(["wind_direction_bin", "beaufort"], observed=False).size().reset_index(name="count")  # type: ignore
    df_windrose["frequency"] = df_windrose["count"] / df_windrose["count"].sum() * 100
    df_windrose["wind_direction_bin_degs"] = df_windrose["wind_direction_bin"].cat.rename_categories(
        {
            "N": 0,
            "NNE": 22.5,
            "NE": 45,
            "ENE": 67.5,
            "E": 90,
            "ESE": 112.5,
            "SE": 135,
            "SSE": 157.5,
            "S": 180,
            "SSW": 202.5,
            "SW": 225,
            "WSW": 247.5,
            "W": 270,
            "WNW": 292.5,
            "NW": 315,
            "NNW": 337.5,
        },
    )
    df_windrose["beaufort"] = df_windrose["beaufort"].astype(int)
    return df_windrose


def windrose_graph(df, plot_transect=False, theta1=None, theta2=None):
    n_colors = 13
    colors = px.colors.sample_colorscale("turbo", [n / (n_colors - 1) for n in range(n_colors)])
    fig = px.bar_polar(
        df,
        r="frequency",
        theta="wind_direction_bin_degs",
        color="beaufort",
        labels={
            "frequency": "Frequency (%)",
            "wind_direction_bin": "Direction",
            "beaufort": "Beaufort Scale",
        },
        color_discrete_map=colors,
    )
    fig.update_layout(polar=dict(radialaxis={"visible": False, "showticklabels": False}))
    fig.update_layout(
        polar=dict(
            angularaxis={
                "showgrid": False,
            },
        ),
    )

    fig.update_layout(polar_bargap=0)
    if plot_transect:
        max_freq = df.groupby("wind_direction_bin", observed=False)["frequency"].sum().max()
        fig.add_trace(
            go.Scatterpolar(
                r=[max_freq, max_freq],
                theta=[theta1, theta2],
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                showlegend=False,
            ),
        )
    return fig


def windrose(df: pd.DataFrame, plot_transect=False):
    df_windrose = windrose_process(df)
    if plot_transect:
        theta1, theta2 = processing.bimodal_azimuth(df)
        fig = windrose_graph(df_windrose, plot_transect=plot_transect, theta1=theta1, theta2=theta2)
    else:
        fig = windrose_graph(df_windrose, plot_transect=plot_transect)
    return fig


def outliers(original_data: pd.Series, fence_high: float, fence_low: float):
    outliers = np.array(original_data > fence_high) | (original_data < fence_low)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
    fig.add_trace(px.strip(original_data, color=outliers).data[0], row=1, col=1)
    if sum(outliers) > 0:
        fig.add_trace(px.strip(original_data, color=outliers).data[1], row=1, col=1)
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=-0.5,
                y0=fence_high,
                x1=0.5,
                y1=fence_high,
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=-0.5,
                y0=fence_low,
                x1=0.5,
                y1=fence_low,
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )
    fig.update_traces(offsetgroup=0)

    fig.add_trace(px.scatter(original_data, color=outliers).data[0], row=1, col=2)
    if sum(outliers) > 0:
        fig.add_trace(px.scatter(original_data, color=outliers).data[1], row=1, col=2)
    fig.update_layout(showlegend=False, yaxis_title="Windspeed (ms⁻¹)")

    return fig


def contour_krig(
    df: pd.DataFrame,
    gas: str,
    # array of float 64
    xx: np.ndarray,
    yy: np.ndarray,
    field: np.ndarray,
    cut_ground: bool = False,
    x: str = "x",
    y: str = "height_ato",
) -> go.Figure:
    if np.isnan(field).all():
        return blank_figure()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="markers",
            marker={
                "color": df[f"{gas}_normalised"],
                "colorscale": styling["colorscale"],
                "showscale": True,
                "colorbar": {
                    "title": f"{gas} (ppm)",
                },
            },
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Contour(
            z=field.T,
            x=xx[:, 0],
            y=yy[0, :],
            contours={
                "start": field.min(),
                "end": field.max(),
                "size": (field[~np.isnan(field)].max() - field[~np.isnan(field)].min()) / 21,
            },
            colorscale=styling["colorscale"],
            opacity=0.5,
            showlegend=False,
            showscale=False,
        )
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text="horizontal distance on projected flux plane (m)",
        range=[np.min(xx), np.max(xx)],
        ticks="outside",
        tickwidth=1,
        tickcolor="black",
        ticklen=5,
        nticks=20,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text="height above takeoff (m)",
        range=[np.min(yy), np.max(yy)],
        ticks="outside",
        tickwidth=1,
        tickcolor="black",
        ticklen=5,
        nticks=10,
    )
    if cut_ground:
        resolution = 200  # how many points to interpolate over
        df["ground_elevation_ato"] = df.loc[:, "height_ato"] - df.loc[:, "height_agl"]
        df_sorted = df.dropna(subset=[x, "ground_elevation_ato"]).sort_values(x)
        x_min, x_max = df_sorted[x].min(), df_sorted[x].max()
        x_interp = np.linspace(x_min, x_max, resolution)
        ground_ato_interp = np.interp(x_interp, df_sorted[x], df_sorted["ground_elevation_ato"])
        fig.add_trace(
            go.Scatter(
                x=x_interp,
                y=ground_ato_interp,
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name="Interpolated Ground Level",
            )
        )
    fig.layout.coloraxis.colorbar.title = "Emissions flux (kg⋅m⁻²⋅h⁻¹)"

    return fig


def heatmap_krig(xx: np.ndarray, yy: np.ndarray, field: np.ndarray):
    fig = px.imshow(field.T, x=xx[:, 0], y=yy[0, :], color_continuous_scale=styling["colorscale"], origin="lower")
    fig.layout.coloraxis.colorbar.title = "Emissions flux (kg⋅m⁻²⋅h⁻¹)"
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text="horizontal distance on cylindrical projected flux plane (m)",
        range=[xx.min(), xx.max()],
        ticks="outside",
        tickwidth=1,
        tickcolor="black",
        ticklen=5,
        nticks=20,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text="height above ground level (m)",
        range=[yy.min(), yy.max()],
        ticks="outside",
        tickwidth=1,
        tickcolor="black",
        ticklen=5,
        nticks=10,
    )
    fig.update_layout(coloraxis_colorbar=dict(len=0.25))
    return fig


def create_kml_file(data: pd.DataFrame, output_file: str, column: str, altitudemode: str):
    kml = simplekml.Kml()

    min_value = data[column].min()
    max_value = data[column].max()

    custom_colors = [
        "#008080",
        "#70a494",
        "#b4c8a8",
        "#f6edbd",
        "#edbb8a",
        "#de8a5a",
        "#ca562c",
    ]  # based on plotly geyser
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", custom_colors)

    for _index, row in data.iterrows():
        col_normalized = (row[column] - min) / (max_value - min_value)
        color = mcolors.rgb2hex(cmap(col_normalized))

        pnt = kml.newpoint(coords=[(row["longitude"], row["latitude"], row["height_ato"])], altitudemode=altitudemode)
        pnt.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
        pnt.iconstyle.color = simplekml.Color.rgb(int(color[1:3], 16), int(color[3:5], 16), int(color[5:], 16))
        pnt.iconstyle.scale = 0.6
        pnt.description = f"Concentration: {row[column]} ppm"

    kml.save(output_file)
