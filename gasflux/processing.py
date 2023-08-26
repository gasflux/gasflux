"""Processing function, usually implying some kind of filtering or data transformation."""

from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.odr as odr
from scipy.optimize import least_squares
from scipy.signal import find_peaks


# this returns a bimodal heading from a dataframe. minimum difference is in degrees in case the modes are right next to each other
def bimodal_azimuth(df, heading_col="azimuth_heading", min_altitude=5, min_diff=160):
    df = df[df['altitude'] >= min_altitude]
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
def bimodal_elevation(df, heading_col="elevation_heading", min_altitude=5, max_slope=70):
    df = df[df['altitude'] >= min_altitude]
    data = df[heading_col].to_numpy()
    data = np.abs(data[~np.isnan(data)])
    data = data[data < max_slope]  # to get around the edge case where vertical movements are modal
    hist, edges = np.histogram(data, bins=50)
    max_freq_idx = np.argsort(hist)[::-1][:2]
    mode = edges[max_freq_idx][0]
    return (mode, -mode)


def altitude_transect_splitter(df):
    heights, bin_edges = np.histogram(df["altitude"], bins=40)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peaks, properties = find_peaks(heights)
    bin_centers[peaks]

    slice_edges = (bin_centers[peaks][:-1] + bin_centers[peaks][1:]) / 2
    slice_edges = np.append(df["altitude"].min(), slice_edges)
    slice_edges = np.append(slice_edges, df["altitude"].max())
    fig, ax = plt.subplots()
    ax.stairs(edges=bin_edges, values=heights, fill=True)
    ax.plot(bin_centers[peaks], heights[peaks], "x", color="red")
    ax.vlines(slice_edges, ymin=0, ymax=max(heights), color="red")
    df["slice"] = pd.cut(df["altitude"], bins=slice_edges, labels=False, include_lowest=True)  # type: ignore
    return df, fig


def add_transect_azimuth_switches(df):  # requires headings
    df["transect"] = 0  # split into lines by incrementing based on azimuth heading switches
    df.loc[df["azimuth_heading"].diff().abs() > 150, "transect"] = 1
    df["transect"] = df["transect"].cumsum()
    return df


def heading_filter(  # requires headings to be added already
    df,
    azimuth_filter,
    azimuth_window,
    elevation_filter,
):
    df_unfiltered = df.copy()
    # Filter out any transects where the absolute value of the elevation heading is greater than the elevation filter.
    # This removes data that doesn't fall within a flat altitude transect as it excludes instances where the drone is climbing or descending significantly.
    df_filtered = df_unfiltered[abs(df["elevation_heading"]) < elevation_filter]

    # Compute the two main azimuths (azimuth is the direction along the horizon) of the data.
    azi1, azi2 = bimodal_azimuth(df_filtered)
    print(f"Drone appears to be flying mainly on the headings {azi1:.2f} degrees and {azi2:.2f} degrees")

    # Filter the dataframe based on the azimuth heading. This is done in two steps:
    # First, the function calculates a rolling average of the azimuth heading over a window of azimuth_window size, and compares this with the limits (azi1 +/- azimuth_filter) and (azi2 +/- azimuth_filter).
    # This ensures that only data where the drone is mainly flying in the direction of azi1 or azi2 (within a margin of azimuth_filter degrees) is included.
    df_filtered = df_filtered[
        (df_filtered["azimuth_heading"].rolling(azimuth_window, center=True).mean() < azi1 + azimuth_filter)
        & (df_filtered["azimuth_heading"] > azi1 - azimuth_filter)
        | (df_filtered["azimuth_heading"] < azi2 + azimuth_filter) & (df_filtered["azimuth_heading"] > azi2 - azimuth_filter)
    ]

    return df_filtered, df_unfiltered


# function to return the start and end of the biggest monotonic series of values from a dictionary
def mCount_max(dict):
    poscount = 0
    negcount = 0
    max_pos_count = 0
    max_pos_transect = 0
    max_neg_count = 0
    max_neg_transect = 0
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
            max_pos_transect = i
            pos_start = i - poscount
        elif max_neg_count < negcount:
            max_neg_count = negcount
            max_neg_transect = i
            neg_start = i - negcount
    if max_pos_count > 0 or max_neg_count > 0:
        if max_pos_count >= max_neg_count:
            return pos_start, max_pos_transect
        elif max_pos_count < max_neg_count:
            return neg_start, max_neg_transect
    else:
        return 0, 0


# this function sorts the data into transects based on azimuth switches and then filters to the biggest monotonic series of values
def largest_monotonic_transect_series(df):
    df = add_transect_azimuth_switches(df)  # heading switches
    alt_dict = dict(df.groupby("transect")["altitude"].mean())
    starttransect, endtransect = mCount_max(alt_dict)  # type: ignore
    df = df[(df["transect"] >= starttransect) & (df["transect"] <= endtransect)]  # filter to the biggest monotonic series of values
    print(
        f"Parsed a flight of {endtransect-starttransect} transects from {alt_dict[starttransect]:.0f}m to {alt_dict[endtransect]:.0f}m between the time of {df.index[0]} and {df.index[-1]}"
    )
    return df, starttransect, endtransect


# this function takes a pre-processed input of some transects and returns groups of transects in monotonic sequences according to altitude
# it's advisable to have each group also use the last transect from the previous group
def monotonic_transect_groups(df):
    df = add_transect_azimuth_switches(df)
    alt_dict = dict(df.groupby("transect")["altitude"].mean())

    group_dict = {}
    previous_altitude = None
    current_group = 1
    previous_trend = None
    first_transect_in_series = True

    for transect, altitude in alt_dict.items():
        if previous_altitude is None:
            group_dict[transect] = f'Group_{current_group}'
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
                first_transect_in_series = False  # this logic deals with the case where someone flies up, flies down one transect and then starts flying up again
            group_dict[transect] = f'Group_{current_group}'
            previous_trend = current_trend
        previous_altitude = altitude
    df['group'] = df['transect'].map(group_dict)

    return df, group_dict


# an attempt at better filter for only transects of interest
# chain length is the number of consecutive points to make it a "transect", azimuth and elevation tolerances are in degrees; parallel_std_dev_tolerance is the std in meters of a 10-point sampled deviation from the longest segment
# smoothing window is the number of points to smooth over for the azimuth and elevation headings. it is a median, as these tend to be single point
def remove_non_transects(df, chain_length=70, azimuth_tolerance=10, elevation_tolerance=40, smoothing_window=5):

    def min_angular_diff_deg(x: float, y: float):
        return min(abs(x - y) % 360, (360 - abs(x - y)) % 360)  # deals with circular co-ordinates

    # returns a list of runs that are marked true
    def get_true_runs(mask):
        enumerated_mask = list(enumerate(mask))  # add index numbers to mask
        groups = groupby(enumerated_mask, key=lambda x: x[1])  # group consecutive true/false values
        true_runs = [list(group) for key, group in groups if key]  # retain groups of True values
        return true_runs

    def split_runs_on_azimuth_inversion(df, runs, azimuth_inversion_threshold=90):
        split_runs = []
        for run in runs:
            last_azimuth = df.iloc[run[0][0]]['azimuth_heading']
            current_run = [run[0]]
            for point in run[1:]:
                current_azimuth = df.iloc[point[0]]['azimuth_heading']
                if abs(current_azimuth - last_azimuth) > azimuth_inversion_threshold:
                    split_runs.append(current_run)
                    current_run = [point]
                else:
                    current_run.append(point)
                last_azimuth = current_azimuth
            split_runs.append(current_run)
        return split_runs

    # Create new columns for filtering reasons, initialized with False
    df_removed = df.copy()
    df_removed['filtered_by_azimuth'] = False
    df_removed['filtered_by_elevation'] = False
    df_removed['filtered_by_chain'] = False

    # Apply azimuth filter
    major_azi_headings = bimodal_azimuth(df, heading_col="azimuth_heading")
    major_elev_headings = bimodal_elevation(df, heading_col="elevation_heading")
    # Apply rolling median to azimuth and elevation headings and store them in new columns
    df_removed['smoothed_azimuth_heading'] = df_removed['azimuth_heading'].rolling(smoothing_window, center=True).median().fillna(df_removed['azimuth_heading'])
    df_removed['smoothed_elevation_heading'] = df_removed['elevation_heading'].rolling(smoothing_window, center=True).median().fillna(df_removed['elevation_heading'])
    azimuth_mask = df_removed['smoothed_azimuth_heading'].apply(lambda x: any([min_angular_diff_deg(x, major) <= azimuth_tolerance for major in major_azi_headings]))
    df_removed.loc[~azimuth_mask, 'filtered_by_azimuth'] = True

    # Apply elevation filter
    elevation_mask = df_removed['smoothed_elevation_heading'].apply(lambda x: any([min_angular_diff_deg(x, major) <= elevation_tolerance for major in major_elev_headings]))
    df_removed.loc[~elevation_mask, 'filtered_by_elevation'] = True

    # Combined mask for azimuth and elevation
    mask = azimuth_mask & elevation_mask

    true_runs = get_true_runs(mask)
    split_runs = split_runs_on_azimuth_inversion(df, true_runs)
    chain_mask = [len(run) >= chain_length for run in split_runs]
    chain_filtered_runs = [run for run, filter_by_chain in zip(split_runs, chain_mask) if filter_by_chain]
    chain_filtered_indices = [point[0] for run in chain_filtered_runs for point in run]
    df_removed.loc[df_removed.index.difference(chain_filtered_indices), 'filtered_by_chain'] = True

    filtered_segments = [df.iloc[run[0][0]:run[-1][0] + 1] for run in chain_filtered_runs]

    df_retained = pd.concat(filtered_segments)
    df_removed = df_removed[~df_removed.index.isin(df_retained.index)]

    return df_removed, df_retained


# linear flight path functions
def linear_reg_equation(B, x):
    return B[0] * x + B[1]  # y = mx + c


def flight_odr_fit(df: pd.DataFrame):
    model = odr.Model(linear_reg_equation)
    data = odr.Data(df["utm_easting"], df["utm_northing"])
    odr_instance = odr.ODR(data, model, beta0=[1, 0])  # initial guess of m=1, c=0
    fit = odr_instance.run()

    # add column of distance from linear fit
    m, c = fit.beta
    df = df.assign(distance_from_fit=abs((m * df["utm_easting"] - df["utm_northing"] + c) / np.sqrt(m**2 + 1)))
    return df, fit.beta


# function to take a 3D plane that is near-linear in the xy and turn it into a linear plane with x as distance along the plane and y as depth into the plane
def flatten_linear_plane(df: pd.DataFrame, distance_filter) -> tuple[pd.DataFrame, float]:
    df, coefs2D = flight_odr_fit(df)
    df = df.loc[df["distance_from_fit"] < distance_filter, :]
    df, coefs2D = flight_odr_fit(df)  # re-fit to filtered points
    df = df.loc[df["distance_from_fit"] < distance_filter, :]
    rotation = np.arctan(coefs2D[0])
    df.loc[:, "x"] = (df["utm_easting"] - df["utm_easting"].min()) * np.cos(-rotation) - (
        df["utm_northing"] - df["utm_northing"].min()
    ) * np.sin(-rotation)
    df.loc[:, "y"] = (df["utm_easting"] - df["utm_easting"].min()) * np.sin(-rotation) + (
        df["utm_northing"] - df["utm_northing"].min()
    ) * np.cos(-rotation)
    df.loc[:, "z"] = df.altitude

    plane_angle = (np.pi / 2) - np.arctan(coefs2D[0])
    return df, plane_angle


# function to set the edges of the linear plane based on not the top and bottom transects
def x_filter(df, starttransect, endtransect, x="x", y="z"):
    bb_df = df[(df["transect"] > starttransect) & (df["transect"] < endtransect)]
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


def wind_rel_ground(df, aircraft_u, aircraft_v, wind_u, wind_v):  # u = N, v = E
    u_wind_ground = wind_u - aircraft_u
    v_wind_ground = wind_v - aircraft_v
    windspd_ground = np.sqrt(u_wind_ground**2 + v_wind_ground**2)
    winddir_ground = np.degrees(np.arctan2(u_wind_ground, v_wind_ground) % (2 * np.pi))
    df["windspd_ground"] = windspd_ground
    df["winddir_ground"] = winddir_ground
    return df
