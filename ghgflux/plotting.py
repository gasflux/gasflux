"""various plotting functions mainly based around plotly"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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
    color: str = "ch4",
    x: str = "utm_easting",
    y: str = "utm_northing",
    z: str = "altitude",
):
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, opacity=0.5, color_continuous_scale=styling["colorscale"])
    fig.update_traces(marker_size=4)
    fig.update_traces(
        customdata=df.index,
        hovertemplate="<br>".join(
            [
                "northing: %{x:.2f}",
                "easting: %{y:.2f}",
                "altitude: %{z:.2f}",
                "CH4: %{marker.color:.2f}",
                "Time: %{customdata}",
            ]
        ),
    )
    return fig


def scatter_2d(
    df: pd.DataFrame,
    x: str = "centred_azimuth",
    y: str = "altitude",
    color: str = "ch4_normalised",
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
                "altitude: %{y:.2f}",
                "CH4: %{marker.color:.2f}",
                "Time: %{customdata}",
            ]
        ),
    )

    return fig


def gas_time_series(df: pd.DataFrame, gas: str = "ch4", color: str = "ch4", split=None):
    fig = px.scatter(df, x=df.index, y=df[gas], color=df[color], color_continuous_scale=styling["colorscale"])
    fig.update_traces(marker_size=8)
    if split is not None:
        y_min, y_max = df[gas].min(), df[gas].max()
        fig.update_layout(
            shapes=[
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=split,
                    y0=y_min,
                    x1=split,
                    y1=y_max,
                    line=dict(color="red", width=2),
                )
            ]
        )
    fig.update_traces(opacity=0.5)

    return fig


def baseline_plotting(df: pd.DataFrame, y: str, bkg: np.ndarray):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ymin = df[y].min()
    ymax = df[y].max()
    ylim = [ymin * 0.95, ymax * 1.05]
    y2min = (df[y] - bkg).min()
    y2lim = (y2min, y2min + (ylim[1] - ylim[0]))
    fig.update_yaxes(range=ylim, secondary_y=False, title_text="Sensor CH4 (ppm)")
    fig.update_yaxes(range=y2lim, secondary_y=True, title_text="Normalised CH4 (ppm)")
    fig.add_scatter(x=df.index, y=df[y], opacity=0.5, name="Raw Data")
    fig.add_scatter(x=df.index, y=bkg, mode="lines", name="Fitted Baseline", line=dict(dash="dash"))
    fig.add_scatter(x=df.index, y=df[y] - bkg, yaxis="y2", name="Normalised Data", mode="lines", opacity=0.5)

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
        labels=[key for key in cardinals.keys()],
        right=False,
    )
    df["wind_direction_bin"] = df["wind_direction_bin"].replace({"N1": "N", "N2": "N"})
    df["beaufort"] = pd.cut(
        df["windspeed"],
        bins=[lower for lower, upper in beaufort.values()] + [list(beaufort.values())[-1][1]],
        labels=[key for key in beaufort.keys()],
        right=False,
    )
    df["beaufort_ms"] = df.replace({"beaufort": beaufort_ms})["beaufort"]
    df_windrose = df.groupby(["wind_direction_bin", "beaufort"]).size().reset_index(name="count")
    df_windrose["frequency"] = df_windrose["count"] / df_windrose["count"].sum() * 100
    df_windrose["wind_direction_bin_degs"] = df_windrose["wind_direction_bin"].replace(
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
        }
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
            }
        )
    )

    fig.update_layout(polar_bargap=0)
    if plot_transect:
        max_freq = df.groupby("wind_direction_bin")["frequency"].sum().max()
        fig.add_trace(
            go.Scatterpolar(
                r=[max_freq, max_freq],
                theta=[theta1, theta2],
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                showlegend=False,
            )
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


def contour_krig(df: pd.DataFrame, xx: np.ndarray, yy: np.ndarray, field: np.ndarray, x: str = "x", y: str = "z"):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="markers",
            marker={
                "color": df["ch4_normalised"],
                "colorscale": styling["colorscale"],
                "showscale": True,
                "colorbar": {
                    "title": "CH₄ (ppm)",
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
                "size": field.max() / 21,
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
    return fig


def slice_grid(df):
    fig, ax = plt.subplots(figsize=(20, 10))
    for i in range(df["slice"].max() - df["slice"].min() + 1):  # zero indexed
        df_slice = df[df["slice"] == i]
        ymin = df_slice["altitude"].min()
        ymax = df_slice["altitude"].max()
        x = sorted(df_slice["circumference_distance"].values)
        y = [ymin, ymax]
        xx, yy = np.meshgrid(x, y)
        z = df_slice["ch4_kg_h_m2"].values
        zz = np.array([z, z])
        ax.pcolormesh(
            xx, yy, zz, cmap="viridis", shading="nearest", clim=(df["ch4_kg_h_m2"].min(), df["ch4_kg_h_m2"].max())
        )
        ax.set_ylim(df["altitude"].min(), df["altitude"].max())
    plt.axis("scaled")
    return fig


def create_kml_file(data: pd.DataFrame, output_file: str, column: str, altitudemode: str):
    kml = simplekml.Kml()

    min = data[column].min()
    max = data[column].max()

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

    for index, row in data.iterrows():
        col_normalized = (row[column] - min) / (max - min)
        color = mcolors.rgb2hex(cmap(col_normalized))

        pnt = kml.newpoint(coords=[(row["longitude"], row["latitude"], row["altitude"])], altitudemode=altitudemode)
        pnt.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
        pnt.iconstyle.color = simplekml.Color.rgb(int(color[1:3], 16), int(color[3:5], 16), int(color[5:], 16))
        pnt.iconstyle.scale = 0.6
        pnt.description = f"Concentration: {row[column]} ppm"

    kml.save(output_file)
