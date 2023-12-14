"""Functions specific to each use case that process the data from start to finished products"""

import numpy as np
import pandas as pd

from . import gas, pre_processing, processing


# functions for ABB GGA data
def ABB_GGA_preprocess(df):
    df = df.rename(
        columns={
            "[CH4]_ppm": "ch4",
            "Latitude (degrees)": "latitude",
            "Longitude (degrees)": "longitude",
            "Altitude (m)": "altitude",
            "WindSpeed3D (m/s)": "windspeed",
            "WindDirection (degree)": "winddir",
        }
    )
    df.index = pd.to_datetime(df["SysTime"])
    df = df.dropna()
    df = pre_processing.add_utm(df)
    # pre_processing.data_tests(df) # removed for now as windspeeds are too high
    df_list = {}
    # split df into several sections based on large gaps in timestamps, called df1, df2 etc.
    for i, df in enumerate(np.array_split(df, np.where(np.diff(df.index) > pd.Timedelta("1m"))[0] + 1)):
        df["altitude"] = df["altitude"] - df["altitude"].min()
        df_list[f"df{i + 1}"] = df
    return df_list


# functions for SeekOps data
def SeekOps_pre_process(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = pd.Index([
        "UTCs",
        "Month",
        "Day",
        "Year",
        "LatitudeDD",
        "LongitudeDD",
        "AltitudeGpsM",
        "PressuremBar",
        "TemperatureC",
        "WindAngleMetDegrees",
        "WindSpdMps",
        "MethanePPB",
    ], dtype="str")
    df = pre_processing.timestamp_from_four_columns(df)
    df = df.loc[~df.index.duplicated(keep="first")].copy()
    df.rename(
        columns={
            "LatitudeDD": "latitude",
            "LongitudeDD": "longitude",
            "AltitudeGpsM": "altitude",
            "MethanePPB": "ch4",
            "WindSpdMps": "windspeed",
            "WindAngleMetDegrees": "winddir",
        },
        inplace=True,
    )
    df["ch4"] = df["ch4"].copy() / 1000
    df = pre_processing.add_utm(df)
    pre_processing.data_tests(df)
    return df


def SeekOps_process(df, celsius, millibars):  # after baseline correction
    df["circ_deviation"], df["circ_azimuth"], df_radius = processing.circle_deviation(
        df, "utm_easting", "utm_northing"
    )
    df = df[df["circ_deviation"].between(-df_radius / 10, df_radius / 10)].copy()  # 10% radius tolerance
    df = processing.recentre_azimuth(df, r=df_radius)
    df["x"] = df["circumference_distance"]
    df = gas.methane_flux_column(df, celsius, millibars)
    return df


# functions for Scientific Aviation daprocessingta
def SciAv_pre_process(folder):
    assert len(list(folder.glob("merge.txt"))) == 1, "more than one merge.txt file found"
    df = pd.read_csv(list(folder.glob("merge.txt"))[0])
    df2 = pd.DataFrame()
    # read all raw files and append to df2
    for file in folder.glob("Z*.txt"):
        df_t = pd.read_csv(file, skiprows=1)
        df_t.columns = [
            "Time Since 1970",
            "wU",
            "wV",
            "wW",
            "a_u",
            "a_v",
            "a_w",
            "Pitch",
            "Roll",
            "Yaw",
            "Latitude",
            "Longitude",
            "Altitude",
            "IsFlying?",
            "BB Time",
            "CH4",
            "C2",
            "CO2",
            "V1",
            "V2",
            "V3",
            "V4",
        ]
        df2 = pd.concat([df2, df_t], axis=0)

    df2["time"] = pd.to_datetime(df2["Time Since 1970"], unit="s").dt.round("1ms").dt.strftime("%H:%M:%S.%f")
    df.drop(columns=["Latitude", "Longitude"], inplace=True)
    df["time"] = pd.to_datetime(df["Time(EPOCH)"], unit="s").dt.round("1ms").dt.strftime("%H:%M:%S.%f")
    df["time"] = pd.to_datetime(df["time"])
    df2["time"] = pd.to_datetime(df2["time"])
    df = pd.merge_asof(df, df2, on="time", direction="nearest", tolerance=pd.Timedelta("2s"))
    df["UTC"] = pd.to_datetime(df["Time(MST)"]).dt.tz_localize("MST").dt.tz_convert("UTC")
    df["timestamp"] = pd.to_datetime(
        df["UTC"].dt.strftime("%Y-%m-%d") + " " + pd.to_datetime(df["time"]).dt.time.astype(str)
    )
    df = df.set_index(df["timestamp"], drop=True)
    df.index.name = "Time(UTC)"
    df = df.drop(columns=["Time(MST)", "Time(EPOCH)", "Time3", "time"])
    df.rename(
        columns={
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Wind Speed (m/s)": "windspeed",
            "Wind Dir (deg)": "winddir",
            "Methane(ppm)": "ch4",
            "Ethane": "c2h6",
            "Altitude": "altitude",
        },
        inplace=True,
    )
    df["altitude"] = df["altitude"].copy() - df["altitude"].iloc[0]  # sets altitude to above home
    name = folder.parts[-2] + "_" + folder.parts[-1]
    df = pre_processing.add_utm(df)
    df, outlier_fig = pre_processing.remove_outliers(df=df, name=name, column="windspeed")
    pre_processing.data_tests(df)
    return df, name, outlier_fig


def SciAv_process(
    df,
    celsius,
    millibars,
    odr_distance_filter,
    azimuth_filter,
    azimuth_window,
    elevation_filter,
):
    original_df = df.copy()
    df = processing.heading_filter(
        df,
        azimuth_filter=azimuth_filter,
        azimuth_window=azimuth_window,
        elevation_filter=elevation_filter,
    )
    df, startrow, endrow = processing.monotonic_row_filter(df)
    df, plane_angle = processing.flatten_linear_plane(df, odr_distance_filter)
    df = gas.methane_flux_column(df, celsius, millibars)
    df = processing.x_filter(df, startrow, endrow)
    removed_df = original_df.loc[original_df.index.difference(df.index)]
    return df, removed_df


def SciAv_process_gaussian(
    df,
    celsius,
    millibars,
    odr_distance_filter,
    azimuth_filter,
    azimuth_window,
    elevation_filter,
    rolling_window,
):
    df = processing.add_rows(df)
    df = processing.heading_filter(
        df,
        azimuth_filter=azimuth_filter,
        azimuth_window=azimuth_window,
        elevation_filter=elevation_filter,
        rolling_window=rolling_window,
    )
    df = processing.add_rows(df)
    df = processing.heading_filter(  # do it twice to remove points on the way up and down
        df, azimuth_filter=azimuth_filter, azimuth_window=azimuth_window, elevation_filter=elevation_filter
    )
    df, plane_angle = processing.flatten_linear_plane(df, odr_distance_filter)
    df = gas.methane_flux_column(df, celsius, millibars)
    return df
