"""Processing function, usually implying some kind of filtering or data transformation."""

from matplotlib.figure import Figure
from itertools import groupby
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import odr
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.stats import circmean


def bimodal_azimuth(
    df: pd.DataFrame, heading_col: str = "azimuth_heading", min_altitude: int = 5, min_diff: int = 160
) -> tuple[float, float]:
    """
    Identifies the two most frequent azimuth headings in the dataset, ensuring they are sufficiently
    distinct. Filters data by altitude and removes NaNs before analysis.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        heading_col (str): Column name for azimuth heading data. Default is "azimuth_heading".
        min_altitude (int): Minimum altitude for data to be included. Default is 5.
        min_diff (int): Minimum difference between the two modes. Default is 160.

    Returns:
        tuple: Two modes of the azimuth heading.
    """
    df = df[df["altitude_ato"] >= min_altitude]
    data = df[heading_col].to_numpy()
    data = data[~np.isnan(data)]
    hist, edges = np.histogram(data, bins=50)
    max_freq_idx = np.argsort(hist)[::-1][:2]
    mode1, mode2 = edges[max_freq_idx][0], edges[max_freq_idx][1]

    while np.abs(mode1 - mode2) < min_diff:
        hist[max_freq_idx[1]] = 0
        max_freq_idx = np.argsort(hist)[::-1][:2]
        mode1, mode2 = edges[max_freq_idx][0], edges[max_freq_idx][1]
    assert 160 < np.abs(mode1 - mode2) < 200
    return (mode1, mode2)


# this returns modes of slope from -90 to 90 degrees.
def bimodal_elevation(
    df: pd.DataFrame, heading_col: str = "elevation_heading", min_altitude: int = 5, max_slope: int = 70
) -> tuple[float, float]:
    """
    Identifies the most frequent elevation heading in the dataset, adjusted for vertical movements.
    Filters data by altitude and removes NaNs before analysis.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        heading_col (str): Column name for elevation heading data. Default is "elevation_heading".
        min_altitude (int): Minimum altitude for data to be included. Default is 5.
        max_slope (int): Maximum slope to consider, avoTupleiding vertical movements. Default is 70.

    Returns:
        tuple: Mode of elevation heading and its negative, representing possible ascent/descent angles.
    """
    df = df[df["altitude_ato"] >= min_altitude]
    data = df[heading_col].to_numpy()
    data = np.abs(data[~np.isnan(data)])
    # to get around the edge case where vertical movements are modal
    data = data[data < max_slope]
    hist, edges = np.histogram(data, bins=50)
    max_freq_idx = np.argsort(hist)[::-1][:2]
    mode = edges[max_freq_idx][0]
    return (mode, -mode)


def altitude_transect_splitter(df: pd.DataFrame) -> tuple[pd.DataFrame, Figure]:
    """
    Splits the dataset into altitude-based transects and plots histogram peaks to identify prominent
    altitude ranges.

    Parameters:
        df (pd.DataFrame): The input dataframe containing altitude data.

    Returns:
        tuple: Modified dataframe with transect labels and a figure showing the histogram with peaks.
    """
    heights, bin_edges = np.histogram(df["altitude_ato"], bins=40)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peaks, properties = find_peaks(heights)
    bin_centers[peaks]

    slice_edges = (bin_centers[peaks][:-1] + bin_centers[peaks][1:]) / 2
    slice_edges = np.append(df["altitude_ato"].min(), slice_edges)
    slice_edges = np.append(slice_edges, df["altitude_ato"].max())
    fig, ax = plt.subplots()
    ax.stairs(edges=bin_edges, values=heights, fill=True)
    ax.plot(bin_centers[peaks], heights[peaks], "x", color="red")
    ax.vlines(slice_edges, ymin=0, ymax=max(heights), color="red")
    df["slice"] = pd.cut(df["altitude_ato"], bins=list(slice_edges), labels=False, include_lowest=True)  # type: ignore
    return df, fig


def add_transect_azimuth_switches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies transects based on significant changes in azimuth heading, incrementing a transect
    counter to distinguish different flight paths.

    Parameters:
        df (pd.DataFrame): The input dataframe with azimuth headings.

    Returns:
        pd.DataFrame: The modified dataframe with a new 'transect' column indicating transect IDs.
    """
    df["transect"] = 0
    df.loc[df["azimuth_heading"].diff().abs() > 150, "transect"] = 1
    df["transect"] = df["transect"].cumsum()
    return df


def heading_filter(
    df: pd.DataFrame, azimuth_filter: float, azimuth_window: int, elevation_filter: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters data based on specified azimuth and elevation headings, aiming to isolate transects that
    align with main flight directions. First elevation is filtered to remove significant climbs or descents
    (beware terrain-following flights). Bimodal azimuth headings are calculated and used to filter the
    data based on the main flight directions, with a rolling mean applied (the window) to smooth the data.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        azimuth_filter (float): The tolerance for deviation from the main azimuth headings.
        azimuth_window (int): The window size for rolling mean calculation of azimuth headings.
        elevation_filter (float): The tolerance for deviation from horizontal flight.

    Returns:
        tuple: The filtered dataframe and the original unfiltered dataframe for comparison.
    """
    df_unfiltered = df.copy()
    df_filtered = df_unfiltered[abs(df["elevation_heading"]) < elevation_filter]

    azi1, azi2 = bimodal_azimuth(df_filtered)
    print(f"Drone appears to be flying mainly on the headings {azi1:.2f} degrees and {azi2:.2f} degrees")

    df_filtered = df_filtered[
        (df_filtered["azimuth_heading"].rolling(azimuth_window, center=True).mean() < azi1 + azimuth_filter)
        & (df_filtered["azimuth_heading"] > azi1 - azimuth_filter)
        | (df_filtered["azimuth_heading"] < azi2 + azimuth_filter)
        & (df_filtered["azimuth_heading"] > azi2 - azimuth_filter)
    ]

    return df_filtered, df_unfiltered


def mCount_max(data_dict: dict[int, float]) -> tuple[int, int]:
    """
    Finds the start and end of the longest monotonic sequence in a dictionary of floats, typically used
    to identify a series of continuous altitude measurements. The first and last are retained.

    Parameters:
    data_dict (Dict[int, float]): Dictionary with sequential numeric keys and numeric values representing measures
    such as altitude.

    Returns:
    tuple: Start and end indices of the longest monotonic sequence in the dictionary.
    """
    poscount = 0
    negcount = 0
    max_pos_count = 0
    max_pos_transect = 0
    max_neg_count = 0
    max_neg_transect = 0
    pos_start = 0
    neg_start = 0

    for i in range(1, len(data_dict)):
        if data_dict[i] >= data_dict[i - 1]:
            poscount += 1
            negcount = 0
        elif data_dict[i] < data_dict[i - 1]:
            negcount += 1
            poscount = 0

        if max_pos_count < poscount:
            max_pos_count = poscount
            max_pos_transect = i
            pos_start = i - poscount
        elif max_neg_count < negcount:
            max_neg_count = negcount
            max_neg_transect = i
            neg_start = i - negcount

    if max_pos_count > 0 or max_neg_count > 0:
        if max_pos_count >= max_neg_count:
            return pos_start, max_pos_transect
        else:
            return neg_start, max_neg_transect
    else:
        return 0, 0


def largest_monotonic_transect_series(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Filters the input dataframe to include only the largest continuous series of transects based on
    monotonic altitude changes.

    Parameters:
        df (pd.DataFrame): The input dataframe with transect and altitude information.

    Returns:
        tuple: The filtered dataframe, start transect, and end transect of the largest monotonic series.
    """
    df = add_transect_azimuth_switches(df)  # heading switches
    alt_dict = dict(df.groupby("transect")["altitude_ato"].mean())
    starttransect, endtransect = mCount_max(alt_dict)  # type: ignore
    # filter to the biggest monotonic series of values
    df = df[(df["transect"] >= starttransect) & (df["transect"] <= endtransect)]
    print(
        f"Parsed a flight of {endtransect-starttransect} transects from {alt_dict[starttransect]:.0f}m to \
            {alt_dict[endtransect]:.0f}m between the time of {df.index[0]} and {df.index[-1]}",
    )
    return df, starttransect, endtransect


def monotonic_transect_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    """
    Groups transects into a dict of monotonic transect sequences based on altitude, facilitating analysis of continuous
    flight patterns. Current behaviour is to reuse end transects of previous sequences as the start of the next.

    Parameters:
        df (pd.DataFrame): The input dataframe with transect and altitude information.

    Returns:
        tuple: The dataframe with a new 'group' column indicating the monotonic group ID, and a
        dictionary mapping transects to group IDs.
    """

    df = add_transect_azimuth_switches(df)
    alt_dict = dict(df.groupby("transect")["altitude_ato"].mean())

    group_dict = {}
    previous_altitude = None
    current_group = 1
    previous_trend = None
    first_transect_in_series = True

    for transect, altitude_ato in alt_dict.items():
        if previous_altitude is None:
            group_dict[transect] = f"Group_{current_group}"
        else:
            if altitude_ato == previous_altitude:
                raise ValueError("Error: altitude_ato is the same as the previous transect")
            elif altitude_ato > previous_altitude:
                current_trend = "ascending"
            else:
                current_trend = "descending"

            if current_trend != previous_trend and previous_trend is not None and not first_transect_in_series:
                current_group += 1  # Increment group counter
                first_transect_in_series = True
            if current_trend == previous_trend or previous_trend is None:
                # handles edge case where someone flies up, flies down one transect and then up again (yes really)
                first_transect_in_series = False
            group_dict[transect] = f"Group_{current_group}"
            previous_trend = current_trend
        previous_altitude = altitude_ato
    df["group"] = df["transect"].map(group_dict)

    return df, group_dict


def remove_non_transects(
    df: pd.DataFrame,
    chain_length: int = 70,
    azimuth_tolerance: int = 10,
    elevation_tolerance: int = 40,
    smoothing_window: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters the dataframe to remove segments not matching the criteria for being considered as transects,
    based on azimuth and elevation headings and segment length.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        chain_length (int): Minimum number of consecutive points to be considered a transect. Default is 70.
        azimuth_tolerance (int): Tolerance for deviation from major azimuth headings. Default is 10.
        elevation_tolerance (int): Tolerance for deviation from major elevation headings. Default is 40.
        smoothing_window (int): Window size for rolling median smoothing of headings. Default is 5. Rolling median
            avoids single point errors.

    Returns:
        tuple: Dataframes of removed segments and retained segments that fit transect criteria.
    """

    def min_angular_diff_deg(x: float, y: float):
        # deals with circular co-ordinates
        return min(abs(x - y) % 360, (360 - abs(x - y)) % 360)

    def get_true_runs(mask):
        enumerated_mask = list(enumerate(mask))  # add index numbers to mask
        # group consecutive true/false values
        groups = groupby(enumerated_mask, key=lambda x: x[1])
        # retain groups of True values
        true_runs = [list(group) for key, group in groups if key]
        return true_runs

    def split_runs_on_azimuth_inversion(df, runs, azimuth_inversion_threshold=120):
        split_runs = []
        for run in runs:
            last_azimuth = df.iloc[run[0][0]]["smoothed_azimuth_heading"]
            current_run = [run[0]]
            for point in run[1:]:
                current_azimuth = df.iloc[point[0]]["smoothed_azimuth_heading"]
                if min_angular_diff_deg(current_azimuth, last_azimuth) > azimuth_inversion_threshold:
                    split_runs.append(current_run)
                    current_run = [point]
                else:
                    current_run.append(point)
                last_azimuth = current_azimuth
            split_runs.append(current_run)
        return split_runs

    # Create new columns for filtering reasons, initialized with False
    df_removed = df.copy()
    df_removed["filtered_by_azimuth"] = False
    df_removed["filtered_by_elevation"] = False
    df_removed["filtered_by_chain"] = False

    # Apply azimuth filter
    major_azi_headings = bimodal_azimuth(df, heading_col="azimuth_heading")
    major_elev_headings = bimodal_elevation(df, heading_col="elevation_heading")

    # Apply rolling median to azimuth and elevation headings and store them in new columns
    df_removed["smoothed_azimuth_heading"] = (
        df_removed["azimuth_heading"]
        .rolling(smoothing_window, center=True)
        .apply(lambda x: circmean(x, 360, 0), raw=True)
        .fillna(df_removed["azimuth_heading"], inplace=False)
    )

    df_removed["smoothed_elevation_heading"] = (
        df_removed["elevation_heading"]
        .rolling(smoothing_window, center=True)
        .apply(lambda x: circmean(x, 360, 0), raw=True)
        .fillna(df_removed["elevation_heading"], inplace=False)
    )

    azimuth_mask = df_removed["smoothed_azimuth_heading"].apply(
        lambda x: any([min_angular_diff_deg(x, major) <= azimuth_tolerance for major in major_azi_headings])
    )

    df_removed.loc[~azimuth_mask, "filtered_by_azimuth"] = True

    # Apply elevation filter
    elevation_mask = df_removed["smoothed_elevation_heading"].apply(
        lambda x: any([min_angular_diff_deg(x, major) <= elevation_tolerance for major in major_elev_headings])
    )
    df_removed.loc[~elevation_mask, "filtered_by_elevation"] = True

    # Combined mask for azimuth and elevation
    mask = azimuth_mask & elevation_mask

    true_runs = get_true_runs(mask)
    split_runs = split_runs_on_azimuth_inversion(df_removed, true_runs)
    chain_mask = [len(run) >= chain_length for run in split_runs]
    chain_filtered_runs = [run for run, filter_by_chain in zip(split_runs, chain_mask, strict=False) if filter_by_chain]
    chain_filtered_indices = [point[0] for run in chain_filtered_runs for point in run]
    df_removed.loc[df_removed.index.difference(chain_filtered_indices), "filtered_by_chain"] = True

    filtered_segments = [df_removed.iloc[run[0][0] : run[-1][0] + 1] for run in chain_filtered_runs]

    df_retained = pd.concat(filtered_segments)
    df_removed = df_removed[~df_removed.index.isin(df_retained.index)]

    return df_removed, df_retained


def orthogonal_distance_regression(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray[np.float64, Any]]:
    """
    Perform orthogonal distance regression on the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing "utm_easting" and "utm_northing" columns.

    Returns:
        tuple[pd.DataFrame, np.ndarray[np.float64, Any]]: Updated DataFrame with "distance_from_fit" column
        and the fitted parameters (slope, intercept).
    """

    def linear_reg_equation(B, x):
        return B[0] * x + B[1]  # y = mx + c

    required_columns = ["utm_easting", "utm_northing"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    model = odr.Model(linear_reg_equation)
    data = odr.Data(df["utm_easting"], df["utm_northing"])

    INITIAL_BETA = [1, 0]  # Initial guess of slope=1, intercept=0
    odr_instance = odr.ODR(data, model, beta0=INITIAL_BETA)
    fit = odr_instance.run()

    if fit.stopreason[0] == "Iteration limit reached":
        raise RuntimeError("ODR fitting failed to converge")

    slope, intercept = fit.beta
    df = df.assign(
        distance_from_fit=abs((slope * df["utm_easting"] - df["utm_northing"] + intercept) / np.sqrt(slope**2 + 1))
    )

    return df, fit.beta


def flatten_linear_plane(df: pd.DataFrame, distance_filter: float = 10000) -> tuple[pd.DataFrame, float]:
    """
    Transforms a 3D dataset into a linear plane, focusing on the largest contiguous dataset aligned with
    the plane's primary orientation. The new x coordinate is the distance along the plane, y is the distance
    perpendicular to the plane (useful only for deviation), and z is the altitude.

    Parameters:
        df (pd.DataFrame): Input dataframe with "utm_easting", "utm_northing", and "altitude_ato" columns.
        distance_filter (float): Threshold for filtering points based on their distance from the regression line.

    Returns:
        tuple: Modified dataframe with new 'x', 'y', and 'z' columns representing transformed coordinates,
        and the plane's angle of rotation.
    """
    df, coefs2D = orthogonal_distance_regression(df)
    df = df.loc[df["distance_from_fit"] < distance_filter, :]
    df, coefs2D = orthogonal_distance_regression(df)  # this is intentionally done twice
    df = df.loc[df["distance_from_fit"] < distance_filter, :]
    rotation = np.arctan(coefs2D[0])
    df.loc[:, "x"] = (df["utm_easting"] - df["utm_easting"].min()) * np.cos(-rotation) - (
        df["utm_northing"] - df["utm_northing"].min()
    ) * np.sin(-rotation)
    df.loc[:, "y"] = (df["utm_easting"] - df["utm_easting"].min()) * np.sin(-rotation) + (
        df["utm_northing"] - df["utm_northing"].min()
    ) * np.cos(-rotation)
    df.loc[:, "z"] = df.altitude_ato

    plane_angle = (np.pi / 2) - np.arctan(coefs2D[0])
    return df, plane_angle


# function to set the edges of the linear plane based on not the top and bottom transects
def x_filter(df: pd.DataFrame, starttransect: int, endtransect: int, x: str = "x", y: str = "z") -> pd.DataFrame:
    """
    Filters the dataframe based on x-coordinates, limiting the dataset to points between specified transects.

    Parameters:
        df (pd.DataFrame): The input dataframe with 'x' and 'y' (or 'z') coordinates.
        starttransect (int): The starting transect for filtering.
        endtransect (int): The ending transect for filtering.
        x (str): Column name for the x-coordinate. Default is 'x'.
        y (str): Column name for the y-coordinate. Default is 'z'.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    bb_df = df[(df["transect"] > starttransect) & (df["transect"] < endtransect)]
    df = df[(df[x] >= bb_df[x].min()) & (df[x] <= bb_df[x].max())]
    return df


## Functions for circular transects ##


def circle_deviation(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Calculates the deviation of points from a fitted circle and their azimuth angles.

    Parameters:
        df (pd.DataFrame): The input dataframe containing the data.
        x (str): Column name for the x-coordinate.
        y (str): Column name for the y-coordinate.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: A tuple containing the deviations of each point from the circle,
        their azimuth angles, and the radius of the fitted circle.
    """
    required_columns = [x_col, y_col]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    x = df[x_col]
    y = df[y_col]

    def midhalf(x):  # select middle half of data to avoid edge effects
        return x.iloc[int(len(x) * 1 / 4) : int(len(x) * 3 / 4)]

    x_filter, y_filter = midhalf(x), midhalf(y)

    def func(params):
        xc, yc, r = params
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

    x_m = np.mean(np.array(x_filter))  # initial guess for parameters
    y_m = np.mean(np.array(y_filter))
    r_m = np.mean(np.sqrt((np.array(x_filter) - x_m) ** 2 + (np.array(y_filter) - y_m) ** 2))

    params0 = np.array([x_m, y_m, r_m])
    result = least_squares(func, params0)
    xc, yc, r = result.x

    deviation = np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

    # output azimuth in radians with 0 at north and increasing clockwise thanks to modulos
    azimuth = np.degrees(np.arctan2(x - xc, y - yc) % (2 * np.pi))

    return deviation, azimuth, r


def recentre_azimuth(df: pd.DataFrame, r: float, x: str = "circ_azimuth", y: str = "ch4_normalised") -> pd.DataFrame:
    """
    Recentres the azimuth angles based on the angle of maximum value and computes the distance along the
    circumference of the circle.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        r (float): The radius of the circle.
        x (str): Column name for azimuth angles. Default is 'circ_azimuth'.
        y (str): Column name for the values used to find the maximum. Default is 'ch4_normalised'.

    Returns:
        pd.DataFrame: The modified dataframe with centered azimuth angles and distances along the circumference.
    """

    def azimuth_of_max(df: pd.DataFrame, x: str = "circ_azimuth", y: str = "ch4_normalised") -> float:
        return df.loc[df[y].idxmax()][x]

    centre_azimuth = azimuth_of_max(df, x, y)
    df["centred_azimuth"] = df[x] - centre_azimuth
    df.loc[df["centred_azimuth"] > 180, "centred_azimuth"] -= 360
    df.loc[df["centred_azimuth"] < -180, "centred_azimuth"] += 360
    df["circumference_distance"] = r * np.radians(df["centred_azimuth"])
    df["circumference_distance"] = df["circumference_distance"] - df["circumference_distance"].min()
    return df


# split data into two dataframes, one for each flight, based on the minimum altitude_ato
def splittime(df: pd.DataFrame) -> pd.Timestamp:
    """
    Determines the splitting time based on the minimum altitude in the dataset.

    Parameters:
        df (pd.DataFrame): The input dataframe with an 'altitude_ato' column.

    Returns:
        pd.Timestamp: The index value corresponding to the minimum altitude, used as a splitting point.
    """
    minalt = df["altitude_ato"].min()
    splittime = df[df["altitude_ato"] == minalt].index[0]
    return splittime


def wind_rel_ground(
    df: pd.DataFrame, aircraft_u: np.ndarray, aircraft_v: np.ndarray, wind_u: np.ndarray, wind_v: np.ndarray
) -> pd.DataFrame:
    """
    Calculates wind speed and direction relative to the ground from aircraft and ambient wind vectors.
    u = N, v = E

    Parameters:
        df (pd.DataFrame): The input dataframe.
        aircraft_u (np.ndarray): The northward component of the aircraft's velocity.
        aircraft_v (np.ndarray): The eastward component of the aircraft's velocity.
        wind_u (np.ndarray): The northward component of the wind's velocity.
        wind_v (np.ndarray): The eastward component of the wind's velocity.

    Returns:
        pd.DataFrame: The dataframe updated with ground-relative wind speed and direction.
    """
    u_wind_ground = wind_u - aircraft_u
    v_wind_ground = wind_v - aircraft_v
    windspeed_ground = np.sqrt(u_wind_ground**2 + v_wind_ground**2)
    winddir_ground = np.degrees(np.arctan2(u_wind_ground, v_wind_ground) % (2 * np.pi))
    df["windspeed_ground"] = windspeed_ground
    df["winddir_ground"] = winddir_ground
    return df
