"""Functions specific to each use case that process the data from start to finished products"""

import pandas as pd

from . import gas, pre_processing, processing


# functions for SeekOps data
def SeekOps_pre_process(df):
    df = pre_processing.timestamp_from_four_columns(df)
    df.rename(
        columns={
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Altitude_m": "altitude",
            "Methane_ppb": "ch4",
            "WindSpeed_ms": "windspeed",
            "WindDirection": "winddir",
        },
        inplace=True,
    )
    df["ch4"] = df["ch4"].copy() / 1000
    df = pre_processing.add_utm(df)
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
    # get altitude, lat and lom from df2 by merging on epoch time
    df2["time"] = pd.to_datetime(df2["Time Since 1970"], unit="s").dt.strftime("%H:%M:%S")
    df2 = df2[["time", "Altitude", "Latitude", "Longitude"]]
    df2["time"] = pd.to_datetime(df2["time"])
    df.drop(columns=["Latitude", "Longitude"], inplace=True)
    df["time"] = pd.to_datetime(df["Time(EPOCH)"], unit="s").dt.time.astype(str)
    df["time"] = pd.to_datetime(df["time"])
    df = pd.merge_asof(df, df2, on="time", direction="nearest", tolerance=pd.Timedelta("2s"))
    # reindex to UTC(?) from MST
    df = df.set_index(df["Time(MST)"].apply(lambda x: pd.Timestamp(x, tz="MST").tz_convert("UTC")))
    df.index.name = "Time(UTC)"
    df["epoch"] = df["Time(EPOCH)"].apply(lambda x: pd.Timestamp(x, unit="s", tz="UTC"))
    df["seconds"] = df["epoch"].dt.second + df["epoch"].dt.microsecond / 1e6  # type: ignore
    df = df.set_index(df.index.floor("1min") + pd.to_timedelta(df["seconds"], unit="s"))  # type: ignore
    df = df.drop(columns=["Time(MST)", "Time(EPOCH)", "epoch", "seconds", "Time3", "time"])
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
    assert df.index.is_unique, "duplicate indices found"
    df = pre_processing.add_utm(df)
    return df


def SciAv_process(
    df,
    celsius,
    millibars,
    odr_distance_filter,
    azimuth_filter,
    azimuth_window,
    elevation_filter,
):
    df, startrow, endrow = processing.heuristic_row_filter(
        df,
        azimuth_filter=azimuth_filter,
        azimuth_window=azimuth_window,
        elevation_filter=elevation_filter,
    )
    df, plane_angle = processing.flatten_linear_plane(df, odr_distance_filter)
    df = gas.methane_flux_column(df, celsius, millibars)
    df = processing.x_filter(df, startrow, endrow)
    return df


def SeekOps_process(df, celsius, millibars):  # after baseline correction
    df["circ_deviation"], df["circ_azimuth"], df_radius = processing.circle_deviation(
        df, "utm_easting", "utm_northing"
    )
    df = df[df["circ_deviation"].between(-df_radius / 10, df_radius / 10)].copy()  # 10% radius tolerance
    df = processing.recentre_azimuth(df, r=df_radius)
    df = gas.methane_flux_column(df, celsius, millibars)
    return df
