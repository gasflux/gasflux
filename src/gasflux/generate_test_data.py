"""A script to generate minimum required input data for testing and running the gasflux package."""

import numpy as np
import matplotlib.pyplot as plt
import noise
import yaml
from pathlib import Path
import pandas as pd
from geopy.distance import geodesic
from geopy.point import Point
import gasflux


def load_config() -> dict:
    file_path = Path(__file__).parent / "testdata" / "testconfig.yaml"
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
        self.windspeed_avg = params["windspeed_avg"]
        self.windspeed_rel_std = params["windspeed_rel_std"]
        self.surface_roughness = params["surface_roughness"]
        self.seed = params["seed"]
        self.simplex_octaves = params["simplex_octaves"]
        self.simplex_persistence = params["simplex_persistence"]
        self.simplex_lacunarity = params["simplex_lacunarity"]
        self.wind_reference_height = params["wind_reference_height"]
        self.winddir_avg = params["winddir_avg"]
        self.winddir_std = params["winddir_std"]
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

    def generate_windspeed_map(self):
        z0 = self.surface_roughness
        z_ref = self.wind_reference_height
        u_ref = np.random.normal(self.windspeed_avg, self.windspeed_rel_std)
        self.z = np.linspace(0.1, self.scene_altitude_range[1], self.vertical_pixels)
        self.u = u_ref * np.log(self.z / z0) / np.log(z_ref / z0)
        self.windspeed_map = np.tile(self.u[:, np.newaxis], (1, self.horizontal_pixels))
        noise_map = np.zeros((self.vertical_pixels, self.horizontal_pixels))
        for i in range(self.vertical_pixels):
            for j in range(self.horizontal_pixels):
                x = i / self.vertical_pixels
                y = j / self.horizontal_pixels
                noise_val = noise.pnoise2(
                    x,
                    y,
                    octaves=self.simplex_octaves,
                    persistence=self.simplex_persistence,
                    lacunarity=self.simplex_lacunarity,
                )
                noise_map[i, j] = noise_val
        noise_map_std = np.std(noise_map)
        noise_map_mean = np.mean(noise_map)
        centered_noise_map = noise_map - noise_map_mean
        scaled_noise_map = centered_noise_map * (self.windspeed_rel_std / noise_map_std)
        shifted_noise_map = scaled_noise_map + 1
        self.windspeed_map *= shifted_noise_map

    def generate_winddir_map(self):
        self.winddir_map = np.full((self.vertical_pixels, self.horizontal_pixels), self.winddir_avg, dtype=float)
        noise_map = np.zeros((self.vertical_pixels, self.horizontal_pixels))
        for i in range(self.vertical_pixels):
            for j in range(self.horizontal_pixels):
                x = j / self.horizontal_pixels
                y = i / self.vertical_pixels
                noise_val = noise.pnoise2(
                    x,
                    y,
                    octaves=self.simplex_octaves,
                    persistence=self.simplex_persistence,
                    lacunarity=self.simplex_lacunarity,
                    base=0,
                )
                noise_map[i, j] = noise_val
        noise_map_std = np.std(noise_map)
        noise_map_mean = np.mean(noise_map)
        centered_noise_map = noise_map - noise_map_mean
        scaled_noise_map = centered_noise_map * (self.winddir_std / noise_map_std)
        self.winddir_map += scaled_noise_map
        self.winddir_map = (self.winddir_map + self.transect_azimuth + 90) % 360

    def generate_concentration_map(self, gas: str):
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

    def generate_data(self) -> None:
        self.generate_windspeed_map()
        for gas in self.gases:
            self.generate_concentration_map(gas)
        self.generate_winddir_map()

    def generate_sampling_flight(self) -> None:
        self.end_coords = geodesic(meters=self.transect_length).destination(
            Point(self.start_coords), float(self.transect_azimuth)
        )
        self.df = pd.DataFrame(
            index=range(self.total_points),
            columns=[
                "timestamp",
                "latitude",
                "longitude",
                "x",
                "height_ato",
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

        def geopoints(self):
            start_point = Point(self.start_coords)
            points = [start_point]
            distance_per_step = self.transect_length / self.points_per_transect

            for _ in range(1, self.points_per_transect):
                # Use geodesic to find the next point a certain distance towards the azimuth
                next_point = geodesic(meters=distance_per_step).destination(points[-1], self.transect_azimuth)
                points.append(next_point)
            return points

        points = geopoints(self)
        x_increment = (self.sampling_horizontal_range[1] - self.sampling_horizontal_range[0]) / self.points_per_transect
        x_series = np.linspace(
            self.sampling_horizontal_range[0],
            self.sampling_horizontal_range[0] + x_increment * self.points_per_transect,
            self.points_per_transect,
        )
        latitudes = []
        longitudes = []
        xes = []
        for i in range(0, self.number_of_transects):
            if i % 2 == 0:
                latitudes.extend([point.latitude for point in points])
                longitudes.extend([point.longitude for point in points])
                xes.extend(x_series)
            else:  # invert to allow for proper azimuth switches
                latitudes.extend([point.latitude for point in points[::-1]])
                longitudes.extend([point.longitude for point in points[::-1]])
                xes.extend(x_series[::-1])
        self.df["latitude"] = latitudes
        self.df["longitude"] = longitudes
        self.df["x"] = xes
        altitude_step = (
            self.sampling_altitude_ato_range[1] - self.sampling_altitude_ato_range[0]
        ) / self.number_of_transects

        for i in range(self.number_of_transects):
            start_index = i * self.points_per_transect
            end_index = start_index + self.points_per_transect
            self.df.loc[start_index : end_index - 1, "height_ato"] = (
                self.sampling_altitude_ato_range[0] + i * altitude_step
            )
        self.df["temperature"] = self.temperature
        self.df["pressure"] = self.pressure

        def sample_data(self):
            self.df["norm_x"] = (self.df["x"] - self.scene_horizontal_range[0]) / (
                self.scene_horizontal_range[1] - self.scene_horizontal_range[0]
            )
            self.df["norm_altitude_ato"] = (self.df["height_ato"] - self.scene_altitude_range[0]) / (
                self.scene_altitude_range[1] - self.scene_altitude_range[0]
            )
            self.df["index_x"] = np.floor(self.df["norm_x"] * (self.horizontal_pixels - 1)).astype(int)
            self.df["index_y"] = np.floor(self.df["norm_altitude_ato"] * (self.vertical_pixels - 1)).astype(int)
            self.df["index_x"] = np.clip(self.df["index_x"], 0, self.horizontal_pixels - 1)
            self.df["index_y"] = np.clip(self.df["index_y"], 0, self.vertical_pixels - 1)
            self.df["windspeed"] = self.windspeed_map[self.df["index_y"], self.df["index_x"]]
            self.df["winddir"] = self.winddir_map[self.df["index_y"], self.df["index_x"]]
            for gas in self.gases:
                self.df[gas] = self.concentration_maps[gas][self.df["index_y"], self.df["index_x"]]

        sample_data(self)

        self.df = gasflux.pre_processing.add_utm(self.df)
        self.df = gasflux.pre_processing.add_course(self.df)

        self.df_min = self.df.copy()
        retained_columns = [
            "timestamp",
            "latitude",
            "longitude",
            "height_ato",
            "windspeed",
            "winddir",
            "temperature",
            "pressure",
        ]
        for gas in self.gases:
            retained_columns.append(gas)
        self.df_min = self.df_min[retained_columns]

    def plot_data(self, logwind: bool, windspeed: bool, winddir: bool, gas: bool, scatter_3d: bool) -> None:
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
                self.windspeed_map,
                cmap="viridis",
                origin="lower",
                extent=(0, self.scene_horizontal_range[1], 0, self.scene_altitude_range[1]),
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
                self.winddir_map,
                cmap="hsv",
                origin="lower",
                extent=(0, self.scene_horizontal_range[1], 0, self.scene_altitude_range[1]),
            )
            ax4.set_title("Wind Direction Map")
            ax4.set_xlabel("X (m)")
            ax4.set_ylabel("Y (m)")
            cbar = fig.colorbar(im, ax=ax4)
            cbar.set_label("Wind Direction (Degrees)")
            plt.show()
        if scatter_3d:
            fig = gasflux.plotting.scatter_3d(self.df, color="ch4")
            fig.show()


def main():
    data = SimulatedData2D()
    data.generate_data()
    data.generate_sampling_flight()
    data.plot_data(logwind=False, windspeed=False, winddir=False, gas=False, scatter_3d=True)
    data.df_min.to_csv(Path(__file__).parent / "testdata" / "exampledata.csv", index=False)
    data.df.to_csv(Path(__file__).parent / "testdata" / "testdata.csv", index=False)


if __name__ == "__main__":
    main()
