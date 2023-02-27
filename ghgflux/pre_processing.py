"""Functions that organise the data into standard columns in pandas dataframes. Conversion functions (e.g. WGS84 to UTM) are here but transformations take place in processing.py"""

import geopandas as gpd
import pandas as pd


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
    df["utm_easting"] = gdf.geometry.x  # type: ignore
    df["utm_northing"] = gdf.geometry.y  # type: ignore
    return df
