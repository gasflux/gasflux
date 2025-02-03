import logging
from datetime import datetime
from pathlib import Path
from scipy import stats

import pandas as pd
import plotly.graph_objects as go
import yaml

import gasflux

from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_csv(file_path: Path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.exception(f"File not found: {file_path}")
        raise


def load_config(config_path: Path) -> dict:
    """Load a YAML config file and return a dictionary."""
    try:
        with open(config_path) as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.exception(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.exception(f"Error parsing YAML config: {e}")
        raise


class DataValidator:  # TODO(me): decide whether to move this to preprocessing
    """Validate the data before processing."""

    def __init__(self, df: pd.DataFrame, config: dict) -> None:
        """Initialise the validator with the DataFrame and config."""
        self.df = df
        self.required_cols = config["required_cols"].copy()
        self.required_cols.update(config["gases"])

    def validate(self) -> None:
        """Validate the data."""
        self._check_is_df()
        self._check_cols()
        self._check_dtypes()
        self._check_ranges()
        logger.info("Data validation passed")

    def _check_is_df(self) -> None:
        """Check that the input is a DataFrame."""
        if not isinstance(self.df, pd.DataFrame):
            logging.error("Input data is not a DataFrame.")
            raise ValueError("Input data is not a DataFrame.")

    def _check_cols(self) -> None:
        """Check that the required columns are present."""
        missing_cols = set(self.required_cols) - set(self.df.columns)
        if missing_cols:
            logging.error(f"Missing or mislabelled columns: {missing_cols}")
            raise ValueError(f"Missing or mislabelled columns: {missing_cols}")

    def _check_dtypes(self) -> None:
        """Check that the required columns are of the correct type and do not contain NaN values."""
        for col in self.required_cols:
            if col in self.df:
                if self.df[col].isna().any():
                    logging.error(f"Column '{col}' contains NaN values.")
                    raise ValueError(f"Column '{col}' contains NaN values.")
                if self.df[col].dtype != "float64":
                    logging.error(f"Column '{col}' is not of type 'float64'.")
                    raise ValueError(f"Column '{col}' is not of type 'float64'.")

    def _check_ranges(self) -> None:
        """Check that the required columns are within the specified ranges."""
        for col, (min_val, max_val) in self.required_cols.items():
            if col in self.df.columns and not self.df[col].between(min_val, max_val, inclusive="both").all():
                logging.error(f"Column '{col}' contains values out of range: {min_val} to {max_val}.")
                raise ValueError(f"Column '{col}' contains values out of range: {min_val} to {max_val}.")


class BackgroundStrategy(ABC):
    def __init__(self, data_processor):
        self.data_processor = data_processor

    @abstractmethod
    def process(self):
        pass


class AlgorithmicBaselineStrategy(BackgroundStrategy):
    def process(self):
        logger.info("Applying algorithmic background correction")
        for gas in self.data_processor.gases:
            (
                self.data_processor.df,
                self.data_processor.figs["background"][gas],
                self.data_processor.text[f"background_{gas}"],
            ) = gasflux.background.algorithmic_baseline(
                df=self.data_processor.df,
                gas=gas,
                algorithmic_baseline_settings=self.data_processor.config["algorithmic_baseline_settings"],
            )
        self.data_processor.df_std = self.data_processor.df.copy()


class SensorStrategy(ABC):
    def __init__(self, data_processor):
        self.data_processor = data_processor

    @abstractmethod
    def process(self):
        pass


class InSituSensorStrategy(SensorStrategy):
    def process(self):
        logger.info("Processing in-situ (point) data")
        for gas in self.data_processor.gases:
            self.data_processor.figs["scatter_3d"][gas] = gasflux.plotting.scatter_3d(
                df=self.data_processor.df, color=gas, colorbar_title=f"{gas.upper()} flux (kg/m²/h)"
            )
        if SpatialProcessingStrategy == CurtainSpatialProcessingStrategy:
            self.data_processor.figs["windrose"] = gasflux.plotting.windrose(self.data_processor.df, plot_transect=True)
        else:
            self.data_processor.figs["windrose"] = gasflux.plotting.windrose(self.data_processor.df)
        self.data_processor.figs["wind_timeseries"] = gasflux.plotting.time_series(
            self.data_processor.df, ys=["windspeed", "winddir"]
        )


class SpatialProcessingStrategy(ABC):
    def __init__(self, data_processor):
        self.data_processor = data_processor

    @abstractmethod
    def process(self):
        pass


class CurtainSpatialProcessingStrategy(SpatialProcessingStrategy):
    def process(self):
        logger.info("Applying curtain spatial processing")
        self.data_processor.dfs["original"] = self.data_processor.df.copy()
        self.data_processor.df, self.data_processor.start_transect, self.data_processor.end_transect = (
            gasflux.processing.largest_monotonic_transect_series(self.data_processor.df)
        )
        self.data_processor.dfs["removed"] = self.data_processor.dfs["original"].loc[
            self.data_processor.dfs["original"].index.difference(self.data_processor.df.index)
        ]
        self.data_processor.df, self.data_processor.plane_angle = gasflux.processing.flatten_linear_plane(
            self.data_processor.df
        )
        self.data_processor.df = gasflux.processing.wind_offset_correction(
            self.data_processor.df, self.data_processor.plane_angle
        )
        for gas in self.data_processor.gases:
            self.data_processor.df = gasflux.gas.gas_flux_column(self.data_processor.df, gas)
            self.data_processor.figs["scatter_3d"][gas].add_trace(
                go.Scatter3d(
                    x=self.data_processor.dfs["removed"]["utm_easting"],
                    y=self.data_processor.dfs["removed"]["utm_northing"],
                    z=self.data_processor.dfs["removed"]["height_ato"],
                    mode="markers",
                    marker={"size": 2, "color": "black", "symbol": "circle", "opacity": 0.5},
                )
            )


class SpiralSpatialProcessingStrategy(SpatialProcessingStrategy):
    def process(self):
        logger.info("Applying spiral spatial processing")
        self.data_processor.dfs["original"] = self.data_processor.df.copy()
        # self.data_processor.df, self.data_processor.start_transect, self.data_processor.end_transect = (
        #     gasflux.processing.largest_monotonic_transect_series(self.data_processor.df)

        # no wind offset correction - assume wind is perpendicular to the spiral
        self.data_processor.dfs["removed"] = self.data_processor.dfs["original"].loc[
            self.data_processor.dfs["original"].index.difference(self.data_processor.df.index)
        ]
        (
            self.data_processor.df,
            self.data_processor.circle_radius,
            self.data_processor.circle_center_x,
            self.data_processor.circle_center_y,
        ) = gasflux.processing.circle_deviation(self.data_processor.df, x_col="utm_easting", y_col="utm_northing")
        self.data_processor.df = gasflux.processing.recentre_azimuth(
            self.data_processor.df, r=self.data_processor.circle_radius
        )
        self.data_processor.df["x"] = self.data_processor.df["circumference_distance"]
        for gas in self.data_processor.gases:
            self.data_processor.df = gasflux.gas.gas_flux_column(self.data_processor.df, gas)
            self.data_processor.figs["scatter_3d"][gas].add_trace(
                go.Scatter3d(
                    x=self.data_processor.dfs["removed"]["utm_easting"],
                    y=self.data_processor.dfs["removed"]["utm_northing"],
                    z=self.data_processor.dfs["removed"]["height_ato"],
                    mode="markers",
                    marker={"size": 2, "color": "black", "symbol": "circle", "opacity": 0.5},
                )
            )


class InterpolationStrategy(ABC):
    def __init__(self, data_processor):
        self.data_processor = data_processor

    @abstractmethod
    def process(self):
        pass


class KrigingInterpolationStrategy(InterpolationStrategy):
    def process(self):
        logger.info("Applying kriging interpolation")
        for gas in self.data_processor.gases:
            (
                self.data_processor.output_vars["krig_parameters"][gas],
                self.data_processor.text[f"krig_output_{gas}"],
                self.data_processor.figs["contour"][gas],
                self.data_processor.figs["krig_grid"][gas],
                self.data_processor.figs["semivariogram"][gas],
            ) = gasflux.interpolation.ordinary_kriging(
                df=self.data_processor.df,
                x="x",
                y="height_ato",
                gas=gas,
                ordinary_kriging_settings=self.data_processor.config["ordinary_kriging_settings"],
                **self.data_processor.config["semivariogram_settings"],
            )
            logger.info(f"Kriged {gas}")


class DataProcessor:
    def __init__(self, config: dict, df: pd.DataFrame):
        self.config: dict = config
        self.df: pd.DataFrame = df
        self.gases: list[str] = config["gases"]
        self.processing_time = datetime.now()
        self.figs: dict = {
            "scatter_3d": {},
            "windrose": None,
            "wind_timeseries": None,
            "background": {},
            "contour": {},
            "krig_grid": {},
            "semivariogram": {},
        }
        self.text: dict = {}
        self.output_vars: dict = {"krig_parameters": {}, "std": {}}
        self.dfs: dict = {}
        self.reports: dict = {}

    def strategy_selection(self):
        self.background_strategy: BackgroundStrategy
        if self.config["strategies"]["background"] == "algorithm":
            self.background_strategy = AlgorithmicBaselineStrategy(self)
        self.sensor_strategy: SensorStrategy
        if self.config["strategies"]["sensor"] == "insitu":
            self.sensor_strategy = InSituSensorStrategy(self)
        self.spatial_processing_strategy: SpatialProcessingStrategy
        if self.config["strategies"]["spatial"] == "curtain":
            self.spatial_processing_strategy = CurtainSpatialProcessingStrategy(self)
        if self.config["strategies"]["spatial"] == "spiral":
            self.spatial_processing_strategy = SpiralSpatialProcessingStrategy(self)
        self.interpolation_strategy: InterpolationStrategy
        if self.config["strategies"]["interpolation"] == "kriging":
            self.interpolation_strategy = KrigingInterpolationStrategy(self)

    def process(self):
        self.df = gasflux.pre_processing.add_utm(self.df)
        self.df = gasflux.pre_processing.add_course(self.df)
        DataValidator(self.df, self.config).validate()
        self.background_strategy.process()
        self.sensor_strategy.process()
        self.spatial_processing_strategy.process()
        self.interpolation_strategy.process()

        # Reporting
        for gas in self.gases:
            self.reports[gas] = gasflux.reporting.mass_balance_report(
                krig_params=self.output_vars["krig_parameters"][gas],
                wind_fig=self.figs["wind_timeseries"],
                background_fig=self.figs["background"][gas],
                threed_fig=self.figs["scatter_3d"][gas],
                krig_fig=self.figs["contour"][gas],
                windrose_fig=self.figs["windrose"],
            )

        # Collecting descriptive variables
        self.output_vars["std"]["windspeed"] = self.df["windspeed"].std()
        self.output_vars["std"]["windddir"] = stats.circstd(self.df["winddir"], high=360)
        for gas in self.gases:
            self.output_vars["std"][f"{gas}_background"] = self.df.loc[
                ~self.df[f"{gas}_signal"], f"{gas}_normalised"
            ].std()


def process_main(data_file: Path, config_file: Path) -> None:
    """Main function to run the pipeline."""
    config = load_config(config_file)
    name = data_file.stem
    df = read_csv(data_file)

    processor = DataProcessor(config, df)
    processor.strategy_selection()
    processor.process()
    gasflux.reporting.generate_reports(name, processor, config)
    logger.info("Processing complete")
