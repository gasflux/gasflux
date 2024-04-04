"""A script to generate minimum required input data for the gasflux package."""

import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import yaml
from pathlib import Path
import pandas as pd


def load_config() -> dict:
    file_path = Path(__file__).parent / "test_data_config.yaml"
    with open(file_path) as file:
        return yaml.safe_load(file)


params = load_config()

np.random.seed(params["seed"])  # type: ignore


class SimulatedData2D:
    def __init__(self) -> None:  #
        self.horizontal_pixels = params["horizontal_pixels"]
        self.vertical_pixels = params["vertical_pixels"]
        self.num_plumes = params["num_plumes"]
        self.groupiness = params["groupiness"]
        self.spread = params["spread"]
        self.wind_speed_avg = params["wind_speed_avg"]
        self.wind_speed_std = params["wind_speed_std"]
        self.surface_roughness = params["surface_roughness"]
        self.seed = params["seed"]
        self.wind_noise_percent = params["wind_noise_percent"]
        self.perlin_octaves = params["perlin_octaves"]
        self.perlin_persistence = params["perlin_persistence"]
        self.wind_reference_height = params["wind_reference_height"]
        self.wind_direction_avg = params["wind_direction_avg"]
        self.wind_direction_std = params["wind_direction_std"]
        self.wind_direction_noise_scale = params["wind_direction_noise_scale"]
        self.timestamp = params["timestamp"]
        self.flight_time_seconds = params["flight_time_seconds"]
        self.sample_frequency = params["sample_frequency"]
        self.start_coords = params["start_coords"]
        self.transect_azimuth = params["transect_azimuth"]
        self.sampling_altitude_ato_range = params["sampling_altitude_ato_range"]
        self.sampling_horizontal_range = params["sampling_horizontal_range"]
        self.scene_altitude_range = params["scene_altitude_range"]
        self.scene_horizontal_range = params["scene_horizontal_range"]
        self.sampling_horizontal_range = params["sampling_horizontal_range"]
        self.number_of_transects = params["number_of_transects"]
        self.gases = params["gases"]
        self.temperature = params["temperature"]
        self.pressure = params["pressure"]
        self.pixel_size_x = self.scene_horizontal_range[1] / self.horizontal_pixels  # type: ignore
        self.pixel_size_y = self.scene_altitude_range[1] / self.vertical_pixels  # type: ignore
        self.concentration_maps: dict = {}
        self.total_points = self.flight_time_seconds * self.sample_frequency
        self.transect_length = self.scene_horizontal_range[1] - self.scene_horizontal_range[0]
        self.points_per_transect = self.total_points // self.number_of_transects

    def generate_wind_speed_map(self) -> None:
        z0 = self.surface_roughness
        z_ref = self.wind_reference_height
        u_ref = np.random.normal(self.wind_speed_avg, self.wind_speed_std)
        self.z = np.linspace(0.1, self.scene_altitude_range[1], self.vertical_pixels)
        self.u = u_ref * np.log(self.z / z0) / np.log(z_ref / z0)
        self.wind_map = np.repeat(self.u, self.horizontal_pixels).reshape(self.vertical_pixels, self.horizontal_pixels)
        noise = PerlinNoise(octaves=self.perlin_octaves, seed=self.seed)
        x_coords = np.linspace(0, 1, self.horizontal_pixels)
        y_coords = np.linspace(0, 1, self.vertical_pixels)
        noise_map = np.zeros((self.vertical_pixels, self.horizontal_pixels))
        for i, y in enumerate(y_coords):
            noise_map[i, :] = np.array([noise([x, y]) for x in x_coords])
        noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))
        noise_map = noise_map * self.wind_noise_percent / 100 * np.mean(self.wind_map)
        self.wind_map += noise_map

    def generate_concentration_map(self, gas: str):
        self.concentration_maps[gas] = np.zeros((self.vertical_pixels, self.horizontal_pixels))
        self.concentration_maps[gas] = np.zeros((self.vertical_pixels, self.horizontal_pixels))
        max_concentration = self.gases[gas][1]
        min_concentration = self.gases[gas][0]
        for _ in range(self.num_plumes):
            x_mean = (
                np.random.rand() * self.horizontal_pixels * (1 - self.groupiness)
                + self.horizontal_pixels * self.groupiness / 2
            )
            y_mean = (
                np.random.rand() * self.vertical_pixels * (1 - self.groupiness)
                + self.vertical_pixels * self.groupiness / 2
            )
            x_std = np.random.rand() * self.horizontal_pixels * self.spread + self.horizontal_pixels * 0.05
            y_std = np.random.rand() * self.vertical_pixels * self.spread + self.vertical_pixels * 0.05
            concentration = np.random.uniform(min_concentration, max_concentration)
            x, y = np.meshgrid(np.arange(self.horizontal_pixels), np.arange(self.vertical_pixels))
            gaussian = concentration * np.exp(
                -((x - x_mean) ** 2 / (2 * x_std**2) + (y - y_mean) ** 2 / (2 * y_std**2))
            )
            self.concentration_maps[gas] += gaussian
            self.concentration_maps[gas] = (self.concentration_maps[gas] - np.min(self.concentration_maps[gas])) / (
                np.max(self.concentration_maps[gas]) - np.min(self.concentration_maps[gas])
            )
            self.concentration_maps[gas] = (
                self.concentration_maps[gas] * (max_concentration - min_concentration) + min_concentration
            )

    def generate_wind_direction_map(self) -> None:
        noise = PerlinNoise(octaves=self.perlin_octaves, seed=self.seed)
        x_coords = np.linspace(0, 1, self.horizontal_pixels)
        y_coords = np.linspace(0, 1, self.vertical_pixels)
        direction_noise_map = np.zeros((self.vertical_pixels, self.horizontal_pixels))
        for i, y in enumerate(y_coords):
            direction_noise_map[i, :] = np.array([noise([x, y]) for x in x_coords])
        noise_map_std = np.std(direction_noise_map)
        noise_map_mean = np.mean(direction_noise_map)
        desired_std = self.wind_direction_std
        direction_noise_map = direction_noise_map - noise_map_mean
        direction_noise_map = direction_noise_map * desired_std / noise_map_std
        base_wind_direction = np.ones((self.vertical_pixels, self.horizontal_pixels)) * self.wind_direction_avg
        self.wind_direction_map = base_wind_direction + direction_noise_map

    def generate_data(self) -> None:
        self.generate_wind_speed_map()
        for gas in self.gases:
            self.generate_concentration_map(gas)
        self.generate_wind_direction_map()

    def generate_sampling_flight(self) -> None:
        def calculate_coords(start_coords, azimuth, distance):
            R = 6371000
            bearing = np.radians(azimuth)
            start_lat_rad = np.radians(start_coords[0])
            start_lon_rad = np.radians(start_coords[1])
            end_lat_rad = np.arcsin(
                np.sin(start_lat_rad) * np.cos(distance / R)
                + np.cos(start_lat_rad) * np.sin(distance / R) * np.cos(bearing)
            )
            end_lon_rad = start_lon_rad + np.arctan2(
                np.sin(bearing) * np.sin(distance / R) * np.cos(start_lat_rad),
                np.cos(distance / R) - np.sin(start_lat_rad) * np.sin(end_lat_rad),
            )
            end_lat = np.degrees(end_lat_rad)
            end_lon = np.degrees(end_lon_rad)
            return (end_lat, end_lon)

        self.end_coords = calculate_coords(self.start_coords, self.transect_azimuth, self.scene_horizontal_range[1])

        self.df = pd.DataFrame(
            index=range(self.total_points),
            columns=[
                "timestamp",
                "latitude",
                "longitude",
                "x",
                "altitude_ato",
                "windspeed",
                "temperature",
                "pressure",
            ],
        )
        for gas in self.gases:
            self.df[gas] = np.zeros(self.total_points)
        self.df["timestamp"] = pd.date_range(
            start=self.timestamp, periods=self.total_points, freq=f"{1/self.sample_frequency}s"
        )
        lat_increment = (
            (np.sin(np.radians(self.transect_azimuth)) * self.transect_length) / self.points_per_transect / 111111
        )  # 1 degree ~ 111111 meters
        lon_increment = (
            (np.cos(np.radians(self.transect_azimuth)) * self.transect_length)
            / self.points_per_transect
            / (111111 * np.cos(np.radians(self.start_coords[0])))
        )
        x_increment = (self.sampling_horizontal_range[1] - self.sampling_horizontal_range[0]) / self.points_per_transect

        lat_series = np.linspace(
            self.start_coords[0],
            self.start_coords[0] + lat_increment * self.points_per_transect,
            self.points_per_transect,
        )
        lon_series = np.linspace(
            self.start_coords[1],
            self.start_coords[1] + lon_increment * self.points_per_transect,
            self.points_per_transect,
        )
        x_series = np.linspace(
            self.sampling_horizontal_range[0],
            self.sampling_horizontal_range[0] + x_increment * self.points_per_transect,
            self.points_per_transect,
        )

        altitude_step = (
            self.sampling_altitude_ato_range[1] - self.sampling_altitude_ato_range[0]
        ) / self.number_of_transects

        for i in range(self.number_of_transects):
            start_index = i * self.points_per_transect
            end_index = start_index + self.points_per_transect
            self.df.loc[start_index : end_index - 1, "latitude"] = lat_series
            self.df.loc[start_index : end_index - 1, "longitude"] = lon_series
            self.df.loc[start_index : end_index - 1, "x"] = x_series
            self.df.loc[start_index : end_index - 1, "altitude_ato"] = (
                self.sampling_altitude_ato_range[0] + i * altitude_step
            )
        self.df["temperature"] = self.temperature
        self.df["pressure"] = self.pressure

        def sample_data(self):
            self.df["norm_x"] = (self.df["x"] - self.scene_horizontal_range[0]) / (
                self.scene_horizontal_range[1] - self.scene_horizontal_range[0]
            )
            self.df["norm_altitude_ato"] = (self.df["altitude_ato"] - self.scene_altitude_range[0]) / (
                self.scene_altitude_range[1] - self.scene_altitude_range[0]
            )
            self.df["index_x"] = np.floor(self.df["norm_x"] * (self.horizontal_pixels - 1)).astype(int)
            self.df["index_y"] = np.floor(self.df["norm_altitude_ato"] * (self.vertical_pixels - 1)).astype(int)
            self.df["index_x"] = np.clip(self.df["index_x"], 0, self.horizontal_pixels - 1)
            self.df["index_y"] = np.clip(self.df["index_y"], 0, self.vertical_pixels - 1)
            self.df["windspeed"] = self.wind_map[self.df["index_y"], self.df["index_x"]]
            self.df["winddir"] = self.wind_direction_map[self.df["index_y"], self.df["index_x"]]
            for gas in self.gases:
                self.df[gas] = self.concentration_maps[gas][self.df["index_y"], self.df["index_x"]]

        sample_data(self)

    def plot_data(self, logwind: bool, windspeed: bool, winddir: bool, gas: bool) -> None:
        # Plot the log wind profile
        if logwind:
            fig, ax1 = plt.subplots(figsize=(6, 6))
            ax1.plot(self.z, self.u, "b-")
            ax1.set_ylabel("Wind Speed (m/s)")
            ax1.set_xlabel("Height (m)")
            ax1.set_title("Log Wind Profile")
            ax1.grid(True)
            plt.show()

        # Plot the wind speed map
        if windspeed:
            fig, ax2 = plt.subplots(figsize=(20, 8))
            im = ax2.imshow(
                self.wind_map,
                cmap="viridis",
                origin="lower",
                extent=[0, self.scene_horizontal_range[1], 0, self.scene_altitude_range[1]],
            )
            ax2.set_title("Wind Speed Map")
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            cbar = fig.colorbar(im, ax=ax2)
            cbar.set_label("Wind Speed (m/s)")
            plt.show()

        # Plot the concentration map(s)
        if gas:
            num_gases = len(self.gases)
            fig, axes = plt.subplots(num_gases, 1, figsize=(20, 6))
            fig.suptitle("Gas Concentration Maps", fontsize=16)

            for i, gas_name in enumerate(self.gases):
                im = axes[i].imshow(
                    self.concentration_maps[gas_name],
                    cmap="viridis",
                    origin="lower",
                    extent=[0, self.scene_horizontal_range[1], 0, self.scene_altitude_range[1]],
                )
                axes[i].set_title(f"{gas_name.upper()}")
                axes[i].set_xlabel("X (m)")
                axes[i].set_ylabel("Y (m)")
                cbar = fig.colorbar(im, ax=axes[i])
                cbar.set_label(f"{gas_name.upper()} Concentration (ppm)")

            plt.tight_layout()
            plt.show()

        # Plot the wind direction map
        if winddir:
            fig, ax4 = plt.subplots(figsize=(20, 8))
            im = ax4.imshow(
                self.wind_direction_map,
                cmap="hsv",
                origin="lower",
                extent=[0, self.scene_horizontal_range[1], 0, self.scene_altitude_range[1]],
            )
            ax4.set_title("Wind Direction Map")
            ax4.set_xlabel("X (m)")
            ax4.set_ylabel("Y (m)")
            cbar = fig.colorbar(im, ax=ax4)
            cbar.set_label("Wind Direction (Degrees)")
            plt.show()


def main():
    data = SimulatedData2D()
    data.generate_data()
    data.generate_sampling_flight()
    data.plot_data(logwind=False, windspeed=False, winddir=False, gas=True)
    data.df.to_csv(Path(__file__).parent.parent / "tests" / "data" / "testdata.csv", index=False)


if __name__ == "__main__":
    main()
