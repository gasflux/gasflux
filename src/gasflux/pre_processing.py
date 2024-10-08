"""Functions that organise the data into standard columns in pandas dataframes. Conversion functions (e.g. WGS84 to UTM)
are here but transformations take place in processing.py"""

import geopandas as gpd
import numpy as np
import pandas as pd

from . import plotting
from .processing import circ_median


def data_tests(df: pd.DataFrame):
    assert df["ch4"].min() > 1.6, "ch4 values are too low"
    assert df.index.is_monotonic_increasing, "data is not sorted by time"
    assert df.index.is_unique, "data has duplicate timestamps"
    assert df["ch4"].isna().sum() == 0, "ch4 has missing values"
    assert df["windspeed"].min() >= 0, "windspeed values are negative"
    assert df["windspeed"].max() < 20, "windspeed values are too high"
    if df["windspeed"].max() > 15:
        print("Warning: windspeed is greater than 15 m/s, perhaps due to errors in the data.")


# make timestamp column from UTCs, Month, Day, Year
def timestamp_from_four_columns(df):
    df["Year"] = df["Year"] + 2000
    df["time"] = pd.to_datetime(df["UTCs"], unit="s")
    df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    df["timestamp"] = pd.to_datetime(df["date"].dt.date.astype(str) + " " + df["time"].dt.time.astype(str))
    df.index = df["timestamp"]
    df.drop(
        ["Year", "Month", "Day", "time", "date", "timestamp", "UTCs"],
        axis=1,
        inplace=True,
    )
    return df


# add UTM from latitudes and longitudes
def add_utm(df: pd.DataFrame) -> pd.DataFrame:
    gdf = gpd.GeoDataFrame(  # type: ignore
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"], crs="EPSG:4326"),
    )
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Failed to reproject to a GeoDataFrame")
    gdf["utm_easting"] = gdf.geometry.x
    gdf["utm_northing"] = gdf.geometry.y
    output_df = pd.DataFrame(gdf.drop(columns="geometry"))

    return output_df


# add columns for drone course azimuth and elevation
def add_course(df, rolling_window=1):
    df["hor_distance"] = np.sqrt((df["utm_northing"].diff()) ** 2 + (df["utm_easting"].diff()) ** 2)
    df["vert_distance"] = df["height_ato"].diff()
    df["vert_distance"] = pd.to_numeric(df["vert_distance"], errors="coerce")
    df["hor_distance"] = pd.to_numeric(df["hor_distance"], errors="coerce")
    df["course_azimuth"] = (
        (np.degrees(np.arctan2(df["utm_easting"].diff(), df["utm_northing"].diff())) % 360)
        .rolling(rolling_window)
        .apply(lambda x: circ_median(x), raw=True)
    )
    df["course_elevation"] = (
        np.degrees(np.arctan2(df["vert_distance"], df["hor_distance"]))
        .rolling(rolling_window)
        .apply(lambda x: circ_median(x), raw=True)
    )
    return df


def manual_filtering(dict_dfs: dict, split_times: dict, mask_spans: dict) -> dict:
    filtered_dfs = {}
    for name, df in dict_dfs.items():
        if name in mask_spans:
            for i in range(len(mask_spans[name])):
                df = df.drop(
                    df.between_time(mask_spans[name][i].split(" - ")[0], mask_spans[name][i].split(" - ")[1]).index,
                ).copy()
                filtered_dfs[name] = df.copy()
        if name in split_times:
            split_times[name].append("23:59:59")
            split_times[name].insert(0, "00:00:00")
            for i in range(len(split_times[name]) - 1):
                df2 = df.between_time(split_times[name][i], split_times[name][i + 1]).copy()
                filtered_dfs[name + "_" + str(i)] = df2.copy()
        elif name not in split_times:
            filtered_dfs[name] = df.copy()
    return filtered_dfs


def remove_outliers(df: pd.DataFrame, column: str, name: str):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 3 * iqr
    fence_high = q3 + 3 * iqr
    fig = plotting.outliers(df[column], fence_high, fence_low)
    outliers = df.loc[(df[column] < fence_low) | (df[column] > fence_high)]
    if len(outliers) > 0:
        print(f"{len(outliers)} outliers removed from {name} {column} data")
        # nan for outliers, not row removal
        df.loc[(df[column] < fence_low) | (df[column] > fence_high), column] = float("nan")
    elif len(outliers) == 0:
        print(f"No outliers found in {name} {column} data")

    return df, fig
