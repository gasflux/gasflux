"""Baselining functions"""
import numpy as np
import pandas as pd
import pybaselines as pybs

from . import plotting

# for FABC see Cobas, J., et al. A new general-purpose fully automatic baseline-correction procedure for 1D and 2D NMR data. Journal of Magnetic Resonance, 2006, 183(1), 145-151.


def baseline(
    df: pd.DataFrame,
    algorithm: str,
    y: str = "ch4",
    **kwargs,
):
    df = df.copy()
    if len(df) < 20:
        raise ValueError("Dataframe must contain at least 20 rows")
    index = np.arange(df.index.shape[0])
    baseline_fitter = pybs.Baseline(index, check_finite=False)
    fit = getattr(baseline_fitter, algorithm)
    bkg, params = fit(df[y], **kwargs)
    bkg_points = params["mask"]
    background = (df[y] - bkg)[bkg_points]
    signal = (df[y] - bkg)[~bkg_points]
    df["signal"] = np.invert(bkg_points)
    fig = plotting.baseline_plotting(df, y, bkg, signal)
    output_text = (
        f"Baseline algorithm: {algorithm}\n"
        f"Positive and negative 95% percentile of baseline: {np.percentile(background, 2.5):.2f} ppm, {np.percentile(background, 97.5):.2f} ppm\n"
        f"Mean of baseline: {np.mean(background):.2f} ppm\n"
        f"Minimum and maximum of baseline: {np.min(background):.2f} ppm, {np.max(background):.2f} ppm\n"
        f"Signal points: {len(signal)}; background points: {len(background)}\n"
    )
    df[f"{y}_normalised"] = df[y] - bkg
    return df, fig, output_text
