"""Processing function, usually implying some kind of filtering or data transformation."""

import numpy as np
import pandas as pd
import scipy.odr as odr
from scipy.optimize import least_squares


# add columns for drone bearings
def heading(df):
    df["hor_distance"] = np.sqrt((df["utm_northing"].diff()) ** 2 + (df["utm_easting"].diff()) ** 2)
    df["vert_distance"] = df["altitude"].diff()

    df["elevation_heading"] = np.degrees(np.arctan2(df["vert_distance"], df["hor_distance"]))
    df["azimuth_heading"] = np.degrees(np.arctan2(df["utm_easting"].diff(), df["utm_northing"].diff())) % 360
    return df


# function to return the start and end of the biggest monotonic series of values from a dictionary
def mCount(dict):
    poscount = 0
    negcount = 0
    max_pos_count = 0
    max_pos_row = 0
    max_neg_count = 0
    max_neg_row = 0
    pos_start = 0
    neg_start = 0
    for i in range(1, len(dict)):
        if dict[i] >= dict[i - 1]:
            poscount += 1
            negcount = 0
        elif dict[i] < dict[i - 1]:
            negcount += 1
            poscount = 0
        if max_pos_count < poscount:
            max_pos_count = poscount
            max_pos_row = i
            pos_start = i - poscount
        elif max_neg_count < negcount:
            max_neg_count = negcount
            max_neg_row = i
            neg_start = i - negcount
    if max_pos_count > 0 or max_neg_count > 0:
        if max_pos_count >= max_neg_count:
            return pos_start, max_pos_row
        elif max_pos_count < max_neg_count:
            return neg_start, max_neg_row
    else:
        return 0, 0


def bimodal_azimuth(df):
    data = df["azimuth_heading"].to_numpy()
    hist, edges = np.histogram(data, bins=50)
    max_freq_idx = np.argsort(hist)[::-1][:2]
    mode1, mode2 = edges[max_freq_idx][0], edges[max_freq_idx][1]

    while np.abs(mode1 - mode2) < 160:
        hist[max_freq_idx[1]] = 0
        max_freq_idx = np.argsort(hist)[::-1][:2]
        mode1, mode2 = edges[max_freq_idx][0], edges[max_freq_idx][1]
    assert 160 < np.abs(mode1 - mode2) < 200
    return mode1, mode2


def heuristic_row_filter(
    df,
    azimuth_filter,
    azimuth_window,
    elevation_filter,
):
    df = heading(df)
    df = df[abs(df["elevation_heading"]) < elevation_filter]  # degrees, filters ups and downs
    azi1, azi2 = bimodal_azimuth(df)
    print(f"Drone appears to be flying mainly on the headings {azi1:.2f} degrees and {azi2:.2f} degrees")
    df = df[
        (df["azimuth_heading"].rolling(azimuth_window, center=True).mean() < azi1 + azimuth_filter)
        & (df["azimuth_heading"] > azi1 - azimuth_filter)
        | (df["azimuth_heading"] < azi2 + azimuth_filter) & (df["azimuth_heading"] > azi2 - azimuth_filter)
    ]
    df["row"] = 0  # split into lines by incrementing based on azimuth heading switches
    df.loc[df["azimuth_heading"].diff().abs() > 90, "row"] = 1
    df["row"] = df["row"].cumsum()
    alt_dict = dict(df.groupby("row")["altitude"].mean())
    startrow, endrow = mCount(alt_dict)  # type: ignore
    df = df[(df["row"] >= startrow) & (df["row"] <= endrow)]  # filter to the biggest monotonic series of values
    print(
        f"Parsed a flight of {endrow-startrow} rows from {alt_dict[startrow]:.0f}m to {alt_dict[endrow]:.0f}m between the time of {df.index[0]} and {df.index[-1]}"
    )
    return df, startrow, endrow


# linear flight path functions
def linear_reg_equation(coefs: tuple[float, float], x: pd.Series) -> pd.Series:
    return coefs[0] * x + coefs[1]  # y = mx + c


def flight_odr_fit(df: pd.DataFrame):
    fit = odr.odr(linear_reg_equation, [1, 0], y=df["utm_northing"], x=df["utm_easting"])  # inital guess of m=1, c=0
    # add column of distance from linear fit
    df["distance_from_fit"] = np.sqrt(
        (df["utm_northing"] - linear_reg_equation(fit[0], df["utm_easting"])) ** 2
        + (df["utm_easting"] - df["utm_easting"]) ** 2
    )
    return df, fit[0]


def flatten_linear_plane(df: pd.DataFrame, distance_filter) -> tuple[pd.DataFrame, float]:
    df, coefs2D = flight_odr_fit(df)
    df = df[df["distance_from_fit"] < distance_filter].copy()  # filter to points near the linear fit
    df, coefs2D = flight_odr_fit(df)  # re-fit to filtered points
    df = df[df["distance_from_fit"] < distance_filter].copy()  # filter again to points near the linear fit
    rotation = np.arctan(coefs2D[0])
    df["x"] = (df["utm_easting"] - df["utm_easting"].min()) * np.cos(-rotation) - (
        df["utm_northing"] - df["utm_northing"].min()
    ) * np.sin(-rotation)
    df["y"] = (df["utm_easting"] - df["utm_easting"].min()) * np.sin(-rotation) + (
        df["utm_northing"] - df["utm_northing"].min()
    ) * np.cos(-rotation)
    df["z"] = df.altitude
    plane_angle = (np.pi / 2) - np.arctan(coefs2D[0])
    return df, plane_angle


# function to set the edges of the linear plane based on not the top and bottom rows
def x_filter(df, startrow, endrow, x="x", y="z"):
    bb_df = df[(df["row"] > startrow) & (df["row"] < endrow)]
    df = df[(df[x] >= bb_df[x].min()) & (df[x] <= bb_df[x].max())]
    return df


# circular flight path functions
def circle_deviation(df, x: str, y: str):
    # select middle half of data to avoid edge effects
    x, y = df[x], df[y]

    def midhalf(x):
        return x.iloc[int(len(x) * 1 / 4) : int(len(x) * 3 / 4)]

    x_filter, y_filter = midhalf(x), midhalf(y)

    # fit circle to data
    def func(params):
        xc, yc, r = params
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

    # initial guess for parameters
    x_m = np.mean(x_filter)
    y_m = np.mean(y_filter)
    r_m = np.mean(np.sqrt((x_filter - x_m) ** 2 + (y_filter - y_m) ** 2))

    # fit circle
    params0 = np.array([x_m, y_m, r_m])
    result = least_squares(func, params0)
    xc, yc, r = result.x

    # output deviation from circle for filtering purposes
    deviation = np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

    # output azimuth in radians with 0 at north and increasing clockwise thanks to modulos
    azimuth = np.degrees(np.arctan2(x - xc, y - yc) % (2 * np.pi))

    return deviation, azimuth, r


def azimuth_of_max(df, x: str = "circ_azimuth", y: str = "ch4_normalised"):
    return df.loc[df[y].idxmax()][x]


def recentre_azimuth(df, r: float, x: str = "circ_azimuth", y: str = "ch4_normalised"):
    centre_azimuth = azimuth_of_max(df, x, y)
    df["centred_azimuth"] = df[x] - centre_azimuth
    df.loc[df["centred_azimuth"] > 180, "centred_azimuth"] -= 360
    df.loc[df["centred_azimuth"] < -180, "centred_azimuth"] += 360
    df["circumference_distance"] = r * np.radians(df["centred_azimuth"])
    df["circumference_distance"] = df["circumference_distance"] - df["circumference_distance"].min()
    return df


# split data into two dataframes, one for each flight, based on the minimum altitude
def splittime(df):
    minalt = df["altitude"].min()
    splittime = df[df["altitude"] == minalt].index[0]
    return splittime
