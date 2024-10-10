"""Functions related to kriging and other kinds of interpolation"""

import numpy as np
import pandas as pd
import skgstat as skg
from scipy import integrate

from . import plotting


def simpsonintegrate(array: np.ndarray, x_cell_size: float, y_cell_size: float) -> float:
    """Function to obtain the volume of the krig in kgh⁻¹, i.e. the cut-fill volume
    (negative volumes from background noise are subtracted)."""
    grid = np.nan_to_num(array.copy(), copy=False, nan=0)
    vol_rows = integrate.simpson(np.transpose(grid))  # this integrates along each row of the grid
    vol_grid = integrate.simpson(vol_rows)  # this integrates the rows together
    return vol_grid * x_cell_size * y_cell_size  # type: ignore


def directional_gas_semivariogram(
    df: pd.DataFrame, x: str, z: str, gas: str, semivariogram_filter: float | None = None, **semivariogram_settings
):
    """Function to calculate the directional semivariogram - typically horizontally - of a gas in a dataframe."""
    if semivariogram_filter:
        df = df[df[gas] > semivariogram_filter]
    v = skg.DirectionalVariogram(
        df[[x, z]].to_numpy(),
        df[gas].to_numpy(),
        **semivariogram_settings,
    )
    return v


def ordinary_kriging(
    df: pd.DataFrame,
    x: str,
    y: str,
    gas: str,
    ordinary_kriging_settings: dict,
    semivariogram_filter: float | None = None,
    **semivariogram_settings,
):
    """Function to calculate the ordinary kriging of a gas in a dataframe, after calculating a semivariogram."""
    gasflux = f"{gas}_kg_h_m2"
    skg.plotting.backend("plotly")  # type: ignore
    cut_ground = ordinary_kriging_settings["cut_ground"]
    semivariogram = directional_gas_semivariogram(df, x, y, gasflux, semivariogram_filter, **semivariogram_settings)
    ok = skg.OrdinaryKriging(
        semivariogram,
        coordinates=df[[x, y]].to_numpy(),
        values=df[gasflux].to_numpy(),
        min_points=ordinary_kriging_settings["min_points"],
        max_points=ordinary_kriging_settings["max_points"],
    )
    x_max = df[x].max()
    x_min = df[x].min()
    y_max = df[y].max()
    y_min = df[y].min() if ordinary_kriging_settings["y_min"] is None else ordinary_kriging_settings["y_min"]
    if cut_ground is True:
        df["ground_elevation_ato"] = df.loc[:, "height_ato"] - df.loc[:, "height_agl"]
        y_min = min(df["ground_elevation_ato"].min(), y_min)
    x_range, y_range = x_max - x_min, y_max - y_min
    cell_rough_size = np.sqrt((x_range * y_range) / ordinary_kriging_settings["grid_resolution"])
    x_nodes, y_nodes = (
        max(int(r / cell_rough_size), ordinary_kriging_settings["min_nodes"]) for r in [x_range, y_range]
    )
    x_cell_size = (x_max - x_min) / x_nodes
    y_cell_size = (y_max - y_min) / y_nodes
    xx, yy = np.mgrid[
        x_min : x_max : x_nodes * 1j,
        y_min : y_max : y_nodes * 1j,  # type: ignore
    ]  # type: ignore
    field = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
    if cut_ground:
        field = remove_values_below_ground(df, field, xx, yy)
    volume = simpsonintegrate(field, x_cell_size, y_cell_size)

    fieldpos = np.copy(field)
    fieldpos[fieldpos < 0] = 0
    volumepos = simpsonintegrate(fieldpos, x_cell_size, y_cell_size)

    fieldneg = np.copy(field)
    fieldneg[fieldneg > 0] = 0
    volumeneg = simpsonintegrate(fieldneg, x_cell_size, y_cell_size)

    error_1s = ok.sigma.reshape(xx.shape)
    # np.nan_to_num(error_1s, copy=False, nan=0)
    volume_error = simpsonintegrate(error_1s, x_cell_size, y_cell_size)

    contour_plot = plotting.contour_krig(df=df, gas=gas, xx=xx, yy=yy, field=field, x=x, y=y, cut_ground=cut_ground)
    grid_plot = plotting.heatmap_krig(xx, yy, field)
    output_text = (
        f"The emissions flux of {gas.upper()} is {volume:.3f}kgh⁻¹; "
        f"the cut and fill volumes of the grid are {volumepos:.3f} and {volumeneg:.3f}kgh⁻¹. "
        f"The grid itself is {x_nodes}x{y_nodes} nodes, with nodes measuring {x_cell_size:.2f}m x {y_cell_size:.2f}m."
    )
    krig_variables = {
        "gas": gas,
        "field": field,
        "fieldpos": fieldpos,
        "fieldneg": fieldneg,
        "xx": xx,
        "yy": yy,
        "volume": volume,
        "volumepos": volumepos,
        "volumeneg": volumeneg,
        "error field (1 sigma)": error_1s,
        "volume_error": volume_error,
    }
    semivariogram_plot = semivariogram.plot(show=False)

    return krig_variables, output_text, contour_plot, grid_plot, semivariogram_plot


def remove_values_below_ground(
    df: pd.DataFrame, field: np.ndarray, xx: np.ndarray, yy: np.ndarray, x: str = "x", alt: str = "height_ato"
) -> np.ndarray:
    """
    Adjust field values based on elevation data, setting values below ground to NaN.
    """
    max_x = df[x].max()
    max_y = df[alt].max()

    x_right = np.empty_like(xx)
    x_right[:-1, :] = xx[1:, :]
    x_right[-1, :] = max_x
    x_points = (xx + x_right) / 2

    y_top = np.empty_like(yy)
    y_top[:, :-1] = yy[:, 1:]
    y_top[:, -1] = max_y
    y_points = (yy + y_top) / 2

    x_points_flat = x_points.ravel()
    y_points_flat = y_points.ravel()

    ground_levels = compute_relative_ground_levels(df, x_points_flat)

    below_ground = y_points_flat < ground_levels
    field_flat = field.ravel()
    field_flat[below_ground] = np.nan

    return field_flat.reshape(field.shape)


def compute_relative_ground_levels(
    df: pd.DataFrame, x_points: np.ndarray, y1: str = "height_agl", y2: str = "height_ato"
) -> np.ndarray:
    """
    Calculate ground levels at given x coordinates, considering elevation above ground and takeoff altitude. A
    sort of janky averaged DEM, basically. Will accept either ground elevation and altitude, or height above ground
    level and height above takeoff as inputs for y1 and y2.
    """
    df_clean = df.dropna(subset=[y1, y2])
    df_sorted = df_clean.sort_values(by="x").drop_duplicates(subset="x")
    y1_at_x = np.interp(x_points, df_sorted["x"], df_sorted[y1])
    y2_at_x = np.interp(x_points, df_sorted["x"], df_sorted[y2])

    return y2_at_x - y1_at_x


# def additive_row_integration(df: pd.DataFrame, rowlabel: str = "slice"):
#     """2D integration of a dataframe along the x-axis, with the altitude as the y-axis."""
#     integrals = {}
#     for i in range(df[rowlabel].max() + 1):
#         df_slice = df[df[rowlabel] == i]
#         df_slice = df_slice.sort_values(by="x")
#         line_integral = integrate.simpson(y=df_slice["ch4_kg_h_m2"], x=df_slice["x"])
#         area_integral = line_integral * (df_slice["altitude"].max() - df_slice["altitude"].min())
#         integrals[i] = area_integral
#     return integrals
