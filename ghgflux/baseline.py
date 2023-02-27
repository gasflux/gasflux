"""Baselining functions"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pybaselines as pybs

# for FABC see Cobas, J., et al. A new general-purpose fully automatic baseline-correction procedure for 1D and 2D NMR data. Journal of Magnetic Resonance, 2006, 183(1), 145-151.


def baseline(
    df,
    algorithm: str,
    name=None,
    y: str = "ch4",
    colors: str = "altitude",
    plot: bool = False,
    **kwargs,
):
    df = df.copy()
    index = np.arange(df.index.shape[0])
    baseline_fitter = pybs.Baseline(index, check_finite=False)
    fit = getattr(baseline_fitter, algorithm)
    bkg, params = fit(df[y], **kwargs)
    bkg_points = params["mask"]
    background = (df[y] - bkg)[bkg_points]
    signal = (df[y] - bkg)[~bkg_points]
    if plot:
        x = df.index
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(x, df[y], label="Raw Data", lw=1.5, alpha=0.5)
        ax1.plot(x, bkg, "--", label="Fitted Baseline")

        ax2 = ax1.twinx()
        ax2.set_xlabel("Time")
        ax2.set_ylabel("CH4 (ppm)")

        nml = ax2.scatter(x, df[y] - bkg, label="Normalised Data", c=colors, s=5)
        plt.colorbar(nml, label="Altitude above ground level (m)", pad=0.1)
        y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
        yminax2 = ax2.get_ylim()[0]
        ax2.set_ylim(yminax2, yminax2 + y_range)
        ax2.scatter(
            x[~bkg_points],
            (df[y] - bkg)[~bkg_points],
            label="Methane signal",
            color="red",
            s=7,
        )
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax2.set_ylabel("CH₄ (ppm)")
        ax1.set_ylabel("CH₄ (ppm)")
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.show()
        if name is not None:
            plt.savefig(f"products/{name}_baseline.png")
        else:
            print("No name given for baseline plot, not saving.")
    print(f"Baseline algorithm: {algorithm}")
    print(
        f"Positive and negative 95% percentile of baseline: {np.percentile(background, 2.5):.0f} ppm, {np.percentile(background, 97.5):.0f} ppm"
    )
    print(f"Mean of baseline: {np.mean(background):.2f} ppm")
    print(f"Minimum and maximum of baseline: {np.min(background):.2f} ppm, {np.max(background):.2f} ppm")
    print(f"Signal points: {len(signal)}; background points: {len(background)}")
    df[f"{y}_normalised"] = df[y] - bkg
    return df
