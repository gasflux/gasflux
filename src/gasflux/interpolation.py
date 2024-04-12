"""Functions related to kriging and other kinds of interpolation"""

import numpy as np
import pandas as pd
import skgstat as skg
from scipy import integrate

from . import plotting


def simpsonintegrate(array: np.ndarray, x_cell_size: float, y_cell_size: float) -> float:
    """Function to obtain the volume of the krig in kgh⁻¹, i.e. the cut-fill volume
    (negative volumes from background noise are subtracted)."""
    # this integrates along each row of the grid
    vol_rows = integrate.simpson(np.transpose(array))
    vol_grid = integrate.simpson(vol_rows)  # this integrates the rows together
    return vol_grid * x_cell_size * y_cell_size  # type: ignore


def directional_gas_semivariogram(
    df: pd.DataFrame, x: str, z: str, gas: str, semivariogram_filter: float | None = None, **semivariogram_settings
):
    """Function to calculate the directional semivariogram - typically horizontally - of a gas in a dataframe."""
    if semivariogram_filter:
        df = df[df[gas] > semivariogram_filter]
    v = skg.Variogram(
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
    y_min = df[y].min()
    x_range, y_range = df[x].max() - df[x].min(), df[y].max() - df[y].min()
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

    np.nan_to_num(field, copy=False, nan=0)
    volume = simpsonintegrate(field, x_cell_size, y_cell_size)

    fieldpos = np.copy(field)
    fieldpos[fieldpos < 0] = 0
    volumepos = simpsonintegrate(fieldpos, x_cell_size, y_cell_size)

    fieldneg = np.copy(field)
    fieldneg[fieldneg > 0] = 0
    volumeneg = simpsonintegrate(fieldneg, x_cell_size, y_cell_size)

    error_1s = ok.sigma.reshape(xx.shape)
    np.nan_to_num(error_1s, copy=False, nan=0)
    volume_error = simpsonintegrate(error_1s, x_cell_size, y_cell_size)

    contour_plot = plotting.contour_krig(df=df, gas=gas, xx=xx, yy=yy, field=field, x=x, y=y)
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
