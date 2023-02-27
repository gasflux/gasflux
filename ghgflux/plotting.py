"""various plotting functions mainly based around plotly"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def scatter_3d(
    df: pd.DataFrame,
    name: str | None = None,
    color: str = "ch4",
    x: str = "utm_easting",
    y: str = "utm_northing",
    z: str = "altitude",
):
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, opacity=0.5, color_continuous_scale="geyser")
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
    if name:
        fig.write_html(f"{name}.html")
        fig.update_layout(
            title_text=name,
            title_x=0.5,
            title_font_size=20,
        )
    return fig


def scatter_2d(
    df: pd.DataFrame,
    name: str | None = None,
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
        template="simple_white",
        color_continuous_scale="geyser",
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
    if name:
        fig.write_html(f"{name}.html")
        fig.update_layout(
            title_text=name,
            title_x=0.5,
            title_font_size=20,
        )
    return fig


def gas_time_series_plot(df: pd.DataFrame, name: str, gas: str, color: str, split=None):
    fig = px.scatter(df, x=df.index, y=df[gas], color=df[color])
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
    if name:
        fig.write_html(f"{name}.html")
    fig.show()


def wind_time_series(df, name=str, y=str, color=str):
    fig = px.scatter(df, x=df.index, y=y, color=color, opacity=0.5)
    fig.update_traces(marker_size=8)
    fig.write_html(f"products/{name}.html")
    fig.show()


# plotting windrose
def windrose_process(df):
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
    df["direction_bin"] = pd.cut(
        df["winddir"],
        bins=[lower for lower, upper in cardinals.values()] + [list(cardinals.values())[-1][1]],
        labels=[key for key in cardinals.keys()],
        right=False,
    )
    df["direction_bin"] = df["direction_bin"].replace({"N1": "N", "N2": "N"})
    df["beaufort"] = pd.cut(
        df["windspeed"],
        bins=[lower for lower, upper in beaufort.values()] + [list(beaufort.values())[-1][1]],
        labels=[key for key in beaufort.keys()],
        right=False,
    )
    df_windrose = df.groupby(["direction_bin", "beaufort"]).size().reset_index(name="count")
    df_windrose["frequency"] = df_windrose["count"] / df_windrose["count"].sum() * 100
    df_windrose["direction_degs"] = df_windrose["direction_bin"].replace(
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


def windrose_graph(df, name):
    n_colors = 13
    colors = px.colors.sample_colorscale("turbo", [n / (n_colors - 1) for n in range(n_colors)])
    fig = px.bar_polar(
        df,
        r="frequency",
        theta="direction_bin",
        color="beaufort",
        template="presentation",
        labels={
            "frequency": "Frequency (%)",
            "direction_bin": "Direction",
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
    fig.write_html(f"products/{name}.html")
    fig.show()


def windrose(df, name):
    df_windrose = windrose_process(df)
    windrose_graph(df_windrose, name)


def contour_krig(df: pd.DataFrame, xx: np.ndarray, yy: np.ndarray, field: np.ndarray, x: str, y: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="markers",
            marker={
                "color": df["ch4_kg_h_m2"],
                "colorscale": "geyser",
                "cmin": field.min(),
                "cmax": field.max(),
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
            colorscale="geyser",
            opacity=0.5,
            showlegend=False,
        )
    )
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
    fig.layout.coloraxis.colorbar.title = "Emissions flux (kg⋅m⁻²⋅h⁻¹)"
    fig.update_layout(template="simple_white")
    fig.show()


def heatmap_krig(xx: np.ndarray, yy: np.ndarray, field: np.ndarray):
    fig = px.imshow(field.T, x=xx[:, 0], y=yy[0, :], color_continuous_scale="geyser", origin="lower")
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
    fig.show()
