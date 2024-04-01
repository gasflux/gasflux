"""A script to generate minimum required input data for the gasflux package."""

from math import radians
from pathlib import Path

import numpy as np
import pandas as pd

import gasflux

start_conditions = {
    "timestamp" : "2022-09-26 02:03:00",
    "flight_time_seconds" : 1000,
    "sample_frequency" : 1,
    "start_coords": (54.876670, 15.410000),
    "altitude_ato_range": (-10, 100),  # negative altitudes shouldn't break anything
    "windspeed_range": (4, 10),
    "winddir_range": (90, 120),
    "transect_length": 100,
    "number_of_transects": 10,
    "methane_range": (2.000, 10.000),
    "carbon_dioxide_range": (400.0, 500.0),
    "ethane_range": (0.0, 1.0),
    "temperature": 10.0,
    "pressure": 1000.0,
}

folder = Path(__file__).parent.parent / "tests" / "data"
folder.mkdir(exist_ok=True)


class Test2DDataset:
    def __init__(self, start_conditions):
        self.timestamp = start_conditions.get("timestamp")
        self.flight_time_seconds = start_conditions.get("flight_time_seconds")
        self.sample_frequency = start_conditions.get("sample_frequency")
        self.start_coords = start_conditions.get("start_coords")
        self.altitude_ato_range = start_conditions.get("altitude_ato_range")
        self.windspeed_range = start_conditions.get("windspeed_range")
        self.winddir_range = start_conditions.get("winddir_range")
        self.transect_length = start_conditions.get("transect_length")
        self.number_of_transects = start_conditions.get("number_of_transects")
        self.methane_range = start_conditions.get("methane_range")
        self.carbon_dioxide_range = start_conditions.get("carbon_dioxide_range")
        self.ethane_range = start_conditions.get("ethane_range")
        self.temperature = start_conditions.get("temperature")
        self.pressure = start_conditions.get("pressure")
        self.azimuth = (np.mean(self.winddir_range) + 90) % 360
        self.end_coords = self.calculate_end_coords(self.start_coords, self.azimuth, self.transect_length)
        self.total_points = self.flight_time_seconds * self.sample_frequency
        self.points_per_transect = self.total_points // self.number_of_transects
        self.df = self.generate_position_data()
        self.df = self.generate_sample_data()

    @staticmethod
    def calculate_end_coords(start_coords, azimuth, distance):
        """Calculate end coordinates given start coordinates, azimuth, and distance.
        Assumes a spherical Earth for simplicity.
        """
        R = 6371000  # Earth radius in meters
        bearing = radians(azimuth)

        start_lat_rad = radians(start_coords[0])
        start_lon_rad = radians(start_coords[1])
        end_lat_rad = np.arcsin(np.sin(start_lat_rad) * np.cos(distance / R)
                                + np.cos(start_lat_rad) * np.sin(distance / R) * np.cos(bearing))
        end_lon_rad = start_lon_rad + np.arctan2(np.sin(bearing) * np.sin(distance / R) * np.cos(start_lat_rad),
                                                 np.cos(distance / R) - np.sin(start_lat_rad) * np.sin(end_lat_rad))
        end_lat = np.degrees(end_lat_rad)
        end_lon = np.degrees(end_lon_rad)

        return (end_lat, end_lon)

    def generate_position_data(self):
        np.random.seed(0)
        timestamps = pd.date_range(start=self.timestamp, periods=self.total_points, freq=f"{1/self.sample_frequency}s")
        self.df = pd.DataFrame(index=range(self.total_points), columns=["timestamp", "latitude", "longitude", "altitude_ato", "windspeed", "ch4", "temperature", "pressure"])
        self.df["timestamp"] = timestamps

        lat_increment = (np.sin(np.radians(self.azimuth)) * self.transect_length) / self.points_per_transect / 111111  # 1 degree ~ 111111 meters
        lon_increment = (np.cos(np.radians(self.azimuth)) * self.transect_length) / self.points_per_transect / (111111 * np.cos(np.radians(self.start_coords[0])))

        lat_series = np.linspace(self.start_coords[0], self.start_coords[0] + lat_increment * self.points_per_transect, self.points_per_transect)
        lon_series = np.linspace(self.start_coords[1], self.start_coords[1] + lon_increment * self.points_per_transect, self.points_per_transect)

        altitude_step = (self.altitude_ato_range[1] - self.altitude_ato_range[0]) / self.number_of_transects

        for i in range(self.number_of_transects):
            start_index = i * self.points_per_transect
            end_index = start_index + self.points_per_transect
            self.df.loc[start_index:end_index - 1, "latitude"] = lat_series
            self.df.loc[start_index:end_index - 1, "longitude"] = lon_series
            self.df.loc[start_index:end_index - 1, "altitude_ato"] = self.altitude_ato_range[0] + i * altitude_step

        return self.df

    def generate_sample_data(self):
        np.random.seed(0)
        self.df["windspeed"] = self.df["windspeed"] = np.random.uniform(self.windspeed_range[0], self.windspeed_range[1], self.total_points)
        self.df["ch4"] = np.random.uniform(self.methane_range[0], self.methane_range[1], self.total_points)
        self.df["co2"] = np.random.uniform(self.carbon_dioxide_range[0], self.carbon_dioxide_range[1], self.total_points)
        self.df["c2h6"] = np.random.uniform(self.ethane_range[0], self.ethane_range[1], self.total_points)
        self.df["winddir"] = self.azimuth
        self.df["temperature"] = self.temperature
        self.df["pressure"] = self.pressure
        return self.df


test_data = Test2DDataset(start_conditions)
plot = gasflux.plotting.scatter_3d(df=test_data.df, x="longitude", y="latitude", z="altitude_ato", color="ch4")
plot.show()
test_data.df.to_csv(folder / "testdata.csv", index=False)
