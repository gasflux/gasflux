"""Baselining functions."""

import numpy as np
import pandas as pd
import pybaselines as pybs
from skimage.filters import threshold_triangle as custom_threshold
from . import plotting


def algorithmic_baseline(
    df: pd.DataFrame,
    gas: str,
    algorithmic_baseline_settings: dict,
):
    df = df.copy()
    algorithm = algorithmic_baseline_settings["algorithm"]
    settings = algorithmic_baseline_settings.get(algorithm, {}).copy()
    if settings.get("threshold") == "custom":
        settings["threshold"] = custom_threshold
    if len(df) < 20:
        raise ValueError("Dataframe must contain at least 20 rows for background correction.")
    index = np.arange(len(df))
    baseline_fitter = pybs.Baseline(index, check_finite=False)
    fit = getattr(baseline_fitter, algorithm)
    bkg, params = fit(df[gas], **settings)
    bkg_points = params["mask"]
    df[f"{gas}_normalised"] = df[gas] - bkg
    df[f"{gas}_fit"] = bkg
    background = (df[gas] - bkg)[bkg_points]
    signal = (df[gas] - bkg)[~bkg_points]
    df[f"{gas}_signal"] = np.invert(bkg_points)
    fig = plotting.background_plotting(df, gas)
    output_text = (
        f"Baseline algorithm: {algorithm}\n"
        f"Positive and negative 95% percentile of baseline: {np.percentile(background, 2.5):.2f} ppm, \
            {np.percentile(background, 97.5):.2f} ppm\n"
        f"Mean of baseline: {np.mean(background):.2f} ppm\n"
        f"Minimum and maximum of baseline: {np.min(background):.2f} ppm, {np.max(background):.2f} ppm\n"
        f"Signal points: {len(signal)}; background points: {len(background)}\n"
    )
    return df, fig, output_text
