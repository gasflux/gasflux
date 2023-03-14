"""Functions related top kriging and other kinds of interpolation"""

import numpy as np
import pandas as pd
import skgstat as skg
from scipy import integrate

from . import plotting


def simpsonintegrate(array: np.ndarray, x_cell_size: float, y_cell_size: float) -> float:
    """function to obtain the volume of the krig in kgh⁻¹, i.e. the cut-fill volume (negative volumes from baseline noise are subtracted)."""
    # this integrates along each row of the grid
    vol_rows = integrate.simpson(np.transpose(array))
    vol_grid = integrate.simpson(vol_rows)  # this integrates the rows together
    return vol_grid * x_cell_size * y_cell_size  # type: ignore


def directional_gas_variogram(df: pd.DataFrame, x: str, z: str, gas: str, **variogram_settings):
    v = skg.Variogram(
        df[[x, z]].to_numpy(),
        df[gas].to_numpy(),
        **variogram_settings,
    )
    return v


def ordinary_kriging(
    df: pd.DataFrame,
    x: str,
    y: str,
    gas: str,
    ordinary_kriging_settings: dict,
    **variogram_settings,
):
    skg.plotting.backend("plotly")  # type: ignore
    variogram = directional_gas_variogram(df, x, y, gas, **variogram_settings)
    ok = skg.OrdinaryKriging(
        variogram,
        coordinates=df[[x, y]].to_numpy(),
        values=df[gas].to_numpy(),
        min_points=ordinary_kriging_settings["min_points"],
        max_points=ordinary_kriging_settings["max_points"],
    )
    x_max = df[x].max()
    x_min = df[x].min()
    y_max = df[y].max()
    y_min = df[y].min()
    x_range, y_range = df[x].max() - df[x].min(), df[y].max() - df[y].min()
    cell_rough_size = np.sqrt((x_range * y_range) / ordinary_kriging_settings["grid_resolution"])
    x_nodes, y_nodes = [
        max(int(r / cell_rough_size), ordinary_kriging_settings["min_nodes"]) for r in [x_range, y_range]
    ]
    x_cell_size = (x_max - x_min) / x_nodes
    y_cell_size = (y_max - y_min) / y_nodes
    xx, yy = np.mgrid[
        x_min : x_max : x_nodes * 1j, y_min : y_max : y_nodes * 1j  # type: ignore
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

    s2 = ok.sigma.reshape(xx.shape)
    np.nan_to_num(s2, copy=False, nan=0)
    # volume_error = simpsonintegrate(s2, x_cell_size, y_cell_size)

    contour_plot = plotting.contour_krig(df, xx, yy, fieldpos, x, y)
    grid_plot = plotting.heatmap_krig(xx, yy, fieldpos)
    output_text = (
        f"The emissions flux is {volume:.3f}kgh⁻¹; "
        f"the cut and fill volumes of the grid are {volumepos:.3f} and {volumeneg:.3f}kgh⁻¹. "
        f"The grid itself is {x_nodes}x{y_nodes} nodes, with each node measuring {x_cell_size:.2f}m x {y_cell_size:.2f}m."
    )
    krig_variables = {
        "field": field,
        "fieldpos": fieldpos,
        "fieldneg": fieldneg,
        "xx": xx,
        "yy": yy,
        "volume": volume,
        "volumepos": volumepos,
        "volumeneg": volumeneg,
        "s2": s2,
    }
    variogram_plot = variogram.plot(show=False)

    return krig_variables, output_text, contour_plot, grid_plot, variogram_plot


def additive_row_integration(df: pd.DataFrame, rows: str = "slice"):
    integrals = {}
    for i in range(0, df["slice"].max() + 1):
        df_slice = df[df["slice"] == i]
        df_slice = df_slice.sort_values(by="x")
        line_integral = integrate.simpson(df_slice["ch4_kg_h_m2"], df_slice["x"])
        area_integral = line_integral * (df_slice["altitude"].max() - df_slice["altitude"].min())
        integrals[i] = area_integral
    return integrals
