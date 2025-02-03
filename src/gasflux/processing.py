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
import warnings
import logging


def circ_median(x: np.ndarray | pd.Series) -> float:
    """
    Finds the largest "empty" arc in the circle, splits it there and returns the median of the straight line.
    Probably there are more elegant ways... see https://github.com/scipy/scipy/issues/6644

    Parameters:
        x (np.ndarray | pd.Series): The input array of angles.

    Returns:
        float: The circular median of the input angles.
    """
    angles = sorted(x)
    n = len(angles)
    if n == 1:
        return angles[0]
    gaps = [(angles[(i + 1) % n] - angles[i]) % 360 for i in range(n)]
    largest_gap_index = gaps.index(max(gaps))
    rearranged_angles = angles[largest_gap_index + 1 :] + angles[: largest_gap_index + 1]
    if n % 2 == 1:
        return rearranged_angles[n // 2]
    else:
        return (rearranged_angles[n // 2 - 1] + rearranged_angles[n // 2]) % 360 / 2


def min_angular_displacement(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the minimum circular difference between two angles (in 360 degree space)
    """
    return np.minimum(np.abs(x - y) % 360, (360 - np.abs(x - y)) % 360)


def wind_offset_correction(df: pd.DataFrame, plane_angle: float) -> pd.DataFrame:
    """
    Corrects wind direction data for a given plane angle, assuming the plane is the primary orientation of the dataset.
    This is useful for aligning wind data with the plane's orientation, facilitating analysis of wind effects.

    Parameters:
        df (pd.DataFrame): The input dataframe containing wind direction data.
        wind_dir_col (str): Column name for wind direction data.
        plane_angle (float): Angle of the plane in degrees.

    Returns:
        pd.DataFrame: The modified dataframe with corrected wind direction data.
    """
    df = df.copy()
    df["winddir_rel"] = df.apply(lambda row: abs(90 - min_angular_displacement(row["winddir"], plane_angle)), axis=1)  # type: ignore
    df["windspeed_measured"] = df["windspeed"]
    df["windspeed"] = df["windspeed"] * np.cos(np.radians(df["winddir_rel"]))
    return df


def bimodal_azimuth(
    df: pd.DataFrame, course_col: str = "course_azimuth", min_height: int = 5, min_diff: int = 160
) -> tuple[float, float]:
    """
    Identifies the two most frequent course azimuths in the dataset, ensuring they are sufficiently
    distinct. Filters data by height and removes NaNs before analysis.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        course_col (str): Column name for course azimuth data. Default is "course_azimuth".
        min_altitude (int): Minimum height (_ato typically) for data to be included. Default is 5.
        min_diff (int): Minimum difference between the two modes. Default is 160.

    Returns:
        tuple: Two modes of the course azimuth.
    """
    df = df[df["height_ato"] >= min_height]
    data = df[course_col].dropna().to_numpy()
    hist, edges = np.histogram(data, bins=50)
    edgedist = edges[1] - edges[0]
    bin_centers = edges[:-1] + edgedist / 2
    max_freq_idx = np.argsort(hist)[-2:]  #  top 2 frequencies
    mode1, mode2 = bin_centers[max_freq_idx]
    while np.abs(mode1 - mode2) < min_diff:
        if hist[max_freq_idx[0]] < hist[max_freq_idx[1]]:
            hist[max_freq_idx[0]] = 0
            max_freq_idx = np.argsort(hist)[-2:]
        else:
            hist[max_freq_idx[1]] = 0
            max_freq_idx = np.argsort(hist)[-2:]
        mode1, mode2 = bin_centers[max_freq_idx]
    if min_angular_displacement(mode1, mode2) < 160:
        warnings.warn(
            f"Two modes are close together - this probably should never happen: {mode1:.2f} and {mode2:.2f}",
            UserWarning,
            stacklevel=2,
        )
    return (mode1, mode2)


# this returns modes of slope from -90 to 90 degrees.
def bimodal_elevation(
    df: pd.DataFrame, course_col: str = "course_elevation", min_height: float = 5, max_slope: float = 70
) -> tuple[float, float]:
    """
    Identifies the most frequent course elevation in the dataset, adjusted for vertical movements.
    Filters data by height and removes NaNs before analysis.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        course_col (str): Column name for course elevation data. Default is "course_elevation".
        min_height (int): Minimum height for data to be included. Default is 5.
        max_slope (int): Maximum slope to consider, avoTupleiding vertical movements. Default is 70.

    Returns:
        tuple: Mode of course elevation and its negative, representing possible ascent/descent angles.
    """
    df = df[df["height_ato"] >= df["height_ato"].min() + min_height]
    data = df[course_col].to_numpy()
    data = np.abs(data[~np.isnan(data)])
    # to get around the edge case where vertical movements are modal
    data = data[data < max_slope]
    hist, edges = np.histogram(data, bins=50)
    max_freq_idx = np.argsort(hist)[::-1][:2]
    mode = edges[max_freq_idx][0]
    return (mode, -mode)


def height_transect_splitter(df: pd.DataFrame, height_col: str = "height_ato") -> tuple[pd.DataFrame, Figure]:
    """
    Splits the dataset into height-based transects and plots histogram peaks to identify prominent
    height ranges. Only works if the flights are flat.

    Parameters:
        df (pd.DataFrame): The input dataframe containing height data.

    Returns:
        tuple: Modified dataframe with transect labels and a figure showing the histogram with peaks.
    """
    df = df.copy()
    heights = df[height_col].to_numpy()
    counts, bin_edges = np.histogram(heights, bins=40)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_edges = np.append(heights.min() - bin_width, bin_edges)  # avoid literal edge effects
    bin_edges = np.append(bin_edges, heights.max() + bin_width)
    counts = np.append(0, counts)
    counts = np.append(counts, 0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peaks, properties = find_peaks(counts)
    transect_edges = (bin_centers[peaks][:-1] + bin_centers[peaks][1:]) / 2
    transect_edges = np.append(heights.min(), transect_edges)
    transect_edges = np.append(transect_edges, heights.max())
    fig, ax = plt.subplots()
    ax.stairs(edges=bin_edges, values=counts, fill=True)
    ax.plot(bin_centers[peaks], counts[peaks], "x", color="red")
    ax.vlines(transect_edges, ymin=0, ymax=max(counts), color="red")
    df["transect_num"] = pd.cut(df[height_col], bins=list(transect_edges), labels=False, include_lowest=True)  # type: ignore
    return df, fig


def add_transect_azimuth_switches(df: pd.DataFrame, threshold=150, shift=3) -> pd.DataFrame:
    """
    Identifies transects based on significant changes in course azimuth, incrementing a transect
    counter to distinguish different flight paths. This is a really crude function and should probably
    be only used with data that's already been filtered in some way.

    Parameters:
        df (pd.DataFrame): The input dataframe with course azimuths.

    Returns:
        pd.DataFrame: The modified dataframe with a new 'transect' column indicating transect IDs.
    """
    df = df.copy()
    df["transect_num"] = 0
    df["prev_course_azimuth"] = df["course_azimuth"].shift(shift)  # this gives better behaviour for very neat transects
    df["deg_displace"] = df.apply(
        lambda row: min_angular_displacement(row["course_azimuth"], row["prev_course_azimuth"])  # type: ignore
        if not pd.isnull(row["prev_course_azimuth"])
        else np.nan,
        axis=1,
    )
    df.loc[df["deg_displace"] > threshold, "transect_num"] = 1
    # remove ones that are next to each other
    for i in range(1, len(df)):
        if (df.loc[i, "transect_num"] == 1 and df.loc[i - 1, "transect_num"] == 1) or (
            df.loc[i, "transect_num"] == 1 and df.loc[i - 2, "transect_num"] == 1
        ):
            df.loc[i, "transect_num"] = 0
    df["transect_num"] = df["transect_num"].shift(-shift)  # recorrect for shift
    df["transect_num"] = df["transect_num"].cumsum() + 1  # 1 indexed
    df["transect_num"] = df["transect_num"].ffill()
    df = df.drop(columns=["prev_course_azimuth", "deg_displace"])
    return df


def course_filter(
    df: pd.DataFrame, azimuth_filter: float, azimuth_window: int, elevation_filter: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters data based on specified course azimuth and elevation, aiming to isolate transects that
    align with main flight directions. First elevation is filtered to remove significant climbs or descents
    (beware terrain-following flights). Bimodal course azimuths are calculated and used to filter the
    data based on the main flight directions, with a rolling median applied (the window) to smooth the data.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        azimuth_filter (float): The tolerance for deviation from the main course azimuths.
        azimuth_window (int): The window size for rolling median calculation of course azimuths.
        elevation_filter (float): The tolerance for deviation from horizontal flight.

    Returns:
        tuple: The filtered dataframe and the original unfiltered dataframe for comparison.
    """
    df_filtered = df.copy()
    df_filtered = df_filtered[abs(df["course_elevation"]) < elevation_filter]

    azi1, azi2 = bimodal_azimuth(df_filtered)
    logging.info(f"Drone appears to be flying mainly on the courses {azi1:.2f} degrees and {azi2:.2f} degrees")

    df_filtered["rolling_course_azimuth"] = (
        df_filtered["course_azimuth"].rolling(azimuth_window, center=True).apply(lambda x: circ_median(x), raw=True)
    )

    df_filtered = df_filtered[
        (df_filtered["rolling_course_azimuth"] < azi1 + azimuth_filter)
        & (df_filtered["rolling_course_azimuth"] > azi1 - azimuth_filter)
        | (df_filtered["rolling_course_azimuth"] < azi2 + azimuth_filter)
        & (df_filtered["rolling_course_azimuth"] > azi2 - azimuth_filter)
    ]

    return df_filtered, df


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
    if len(data_dict) < 2:
        raise ValueError("Dictionary must contain at least two values")
    if list(data_dict.keys()) != list(range(1, len(data_dict) + 1)):
        raise ValueError("Keys must be sequential integers starting from 1")
    poscount = 0
    negcount = 0
    max_pos_count = 0
    max_pos_transect = 0
    max_neg_count = 0
    max_neg_transect = 0
    pos_start = 0
    neg_start = 0

    for i in range(2, len(data_dict) + 1):
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


def largest_monotonic_transect_series(
    df: pd.DataFrame, transect_col: str = "transect_num", alt_col: str = "height_ato"
) -> tuple[pd.DataFrame, int, int]:
    """
    Filters the input dataframe to include only the largest continuous series of transects based on
    monotonic altitude changes.

    Parameters:
        df (pd.DataFrame): The input dataframe with transect and altitude information.
        transect_col (str): Column name for transect numbers. Default is "transect_num".
        alt_col (str): Column name for altitude data. Default is "height_ato".

    Returns:
        tuple: The filtered dataframe, start transect, and end transect of the largest monotonic series.
    """
    df = add_transect_azimuth_switches(df)  # course switches
    alt_dict = dict(df.groupby(transect_col)[alt_col].mean())
    starttransect, endtransect = mCount_max(alt_dict)  # type: ignore
    # filter to the biggest monotonic series of values
    df = df[(df[transect_col] >= starttransect) & (df[transect_col] <= endtransect)]
    logging.info(
        f"Parsed a flight of {len(np.unique(df[transect_col]))} transects from {alt_dict[starttransect]:.0f}m"
        f" to {alt_dict[endtransect]:.0f}m between {df['timestamp'].iloc[0]} and {df['timestamp'].iloc[-1]}",
    )
    return df, starttransect, endtransect


def monotonic_transect_groups(
    df: pd.DataFrame, transect_col: str = "transect_num", alt_col: str = "height_ato"
) -> tuple[pd.DataFrame, dict[int, str]]:
    """
    Groups transects into a dict of monotonic transect sequences based on altitude, facilitating analysis of continuous
    flight patterns. Current behaviour is to reuse end transects of previous sequences as the start of the next.

    Parameters:
        df (pd.DataFrame): The input dataframe with transect and altitude information.
        transect_col (str): Column name for transect numbers. Default is "transect_num".
        alt_col (str): Column name for altitude data. Default is "height_ato".

    Returns:
        tuple: The dataframe with a new 'group' column indicating the monotonic group ID, and a
        dictionary mapping transects to group IDs.
    """

    df = add_transect_azimuth_switches(df)
    alt_dict = dict(df.groupby(transect_col)[alt_col].mean())

    group_dict = {}
    previous_altitude = None
    current_group = 1
    previous_trend = None
    first_transect_in_series = True

    for transect, altitude in alt_dict.items():
        if previous_altitude is None:
            group_dict[transect] = f"Group_{current_group}"
        else:
            if altitude == previous_altitude:
                raise ValueError("Error: altitude is the same as the previous transect")
            elif altitude > previous_altitude:
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
        previous_altitude = altitude
    df["group"] = df["transect_num"].map(group_dict)

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
    based on course azimuth and elevation and segment length.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        chain_length (int): Minimum number of consecutive points to be considered a transect. Default is 70.
        azimuth_tolerance (int): Tolerance for deviation from modal course azimuths. Default is 10.
        elevation_tolerance (int): Tolerance for deviation from modal course elevation. Default is 40.
        smoothing_window (int): Window size for rolling median smoothing of courses. Default is 5. Rolling median
            avoids single point errors.

    Returns:
        tuple: Dataframes of removed segments and retained segments that fit transect criteria.
    """

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
            last_azimuth = df.iloc[run[0][0]]["smoothed_course_azimuth"]
            current_run = [run[0]]
            for point in run[1:]:
                current_azimuth = df.iloc[point[0]]["smoothed_course_azimuth"]
                if min_angular_displacement(current_azimuth, last_azimuth) > azimuth_inversion_threshold:
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
    modal_course_azimuths = bimodal_azimuth(df, course_col="course_azimuth")
    modal_elevation_azimuths = bimodal_elevation(df, course_col="course_elevation")

    # Apply rolling median to course azimuth and elevation and store them in new columns
    df_removed["smoothed_course_azimuth"] = (
        df_removed["course_azimuth"]
        .rolling(smoothing_window, center=True)
        .apply(lambda x: circmean(x, 360, 0), raw=True)
        .fillna(df_removed["course_azimuth"], inplace=False)
    )

    df_removed["smoothed_course_elevation"] = (
        df_removed["course_elevation"]
        .rolling(smoothing_window, center=True)
        .apply(lambda x: circmean(x, 360, 0), raw=True)
        .fillna(df_removed["course_elevation"], inplace=False)
    )

    azimuth_mask = df_removed["smoothed_course_azimuth"].apply(
        lambda x: any([min_angular_displacement(x, mode) <= azimuth_tolerance for mode in modal_course_azimuths])
    )

    df_removed.loc[~azimuth_mask, "filtered_by_azimuth"] = True

    # Apply elevation filter
    elevation_mask = df_removed["smoothed_course_elevation"].apply(
        lambda x: any([min_angular_displacement(x, mode) <= elevation_tolerance for mode in modal_elevation_azimuths])
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


def flatten_linear_plane(
    df: pd.DataFrame, alt_col: str = "height_ato", distance_filter: float = 10000
) -> tuple[pd.DataFrame, float]:
    """
    Transforms a 3D dataset into a linear plane, focusing on the largest contiguous dataset aligned with
    the plane's primary orientation. The new x coordinate is the distance along the plane, y is the distance
    perpendicular to the plane (useful only for deviation), and z is the altitude.

    Parameters:
        df (pd.DataFrame): Input dataframe with "utm_easting", "utm_northing", and "height_ato" columns.
        alt_col (str): Column name for altitude data. Default is "height_ato".
        distance_filter (float): Threshold for filtering points based on their distance from the regression line.

    Returns:
        tuple: Modified dataframe with new 'x', 'y', and 'z' columns representing transformed coordinates,
        and one of the plane's angles of rotation (from N, 0-360) in degrees.
    """

    def orthogonal_distance_regression(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray[Any, Any]]:
        """
        Perform orthogonal distance regression on the given DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing "utm_easting" and "utm_northing" columns.

        Returns:
            tuple[pd.DataFrame, np.ndarra[Any,Any]]]: Updated DataFrame with "distance_from_fit" column
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
    df.loc[:, "z"] = df[alt_col]

    plane_angle = (np.pi / 2) - np.arctan(coefs2D[0])
    plane_angle = np.degrees(plane_angle)
    return df, plane_angle


## Functions for circular/spiral flights ##


def circle_deviation(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[pd.DataFrame, float, float, float]:
    """
    Calculates the deviation of points from a fitted circle and their azimuth angles.

    Parameters:
        df (pd.DataFrame): The input dataframe containing the data.
        x (str): Column name for the x-coordinate.
        y (str): Column name for the y-coordinate.

    Returns:
        Tuple[pd.Dataframe, float, float, float]: A tuple containing the modified dataframe with azimuth angles
        and deviations,the radius of the fitted circle, and the coordinates of the circle's center.
    """
    df = df.copy()
    required_columns = [x_col, y_col]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

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

    df["circ_azimuth"] = azimuth
    df["circ_deviation"] = deviation

    return df, r, xc, yc


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
    df = df.copy()

    def azimuth_of_max(df: pd.DataFrame, x: str = "circ_azimuth", y: str = "ch4_normalised") -> float:
        return df.loc[df[y].idxmax()][x]  # type: ignore

    centre_azimuth = azimuth_of_max(df, x, y)
    df["centred_azimuth"] = df[x] - centre_azimuth
    df.loc[df["centred_azimuth"] > 180, "centred_azimuth"] -= 360
    df.loc[df["centred_azimuth"] < -180, "centred_azimuth"] += 360
    df["circumference_distance"] = r * np.radians(df["centred_azimuth"])
    df["circumference_distance"] = df["circumference_distance"] - df["circumference_distance"].min()
    return df


def drone_anemo_to_point_wind(
    df: pd.DataFrame, yaw_col: str, anemo_u_col: str, anemo_v_col: str, easting_col: str, northing_col: str
) -> pd.DataFrame:
    """
    Convert anemometer wind data from drone's coordinate system to Earth's coordinate system
    and calculate true wind speed and direction.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing drone yaw, anemometer data, and drone speed.
    yaw_col (str): Column name for drone's yaw (in degrees, range [-180, 180]).
    anemo_u_col (str): Column name for anemometer U (wind speed in drone's X direction, from port to starboard).
    anemo_v_col (str): Column name for anemometer V (wind speed in drone's Y direction, from aft to nose).
    easting_col (str): Column name for drone's speed from west to east
    northing_col (str): Column name for drone's speed from south to north

    Returns:
    pd.DataFrame: DataFrame with calculated true wind speed ("windspeed") and true wind direction ("winddir").
    """
    yaw_rad = np.deg2rad(df[yaw_col] % 360)
    rotated_U = df[anemo_u_col] * np.cos(yaw_rad) + df[anemo_v_col] * np.sin(yaw_rad)
    rotated_V = -df[anemo_u_col] * np.sin(yaw_rad) + df[anemo_v_col] * np.cos(yaw_rad)
    true_U = -rotated_U - df[easting_col]
    true_V = -rotated_V - df[northing_col]
    df["windspeed"] = np.sqrt(true_U**2 + true_V**2)
    df["winddir"] = np.degrees(np.arctan2(true_U, true_V)) % 360

    return df
