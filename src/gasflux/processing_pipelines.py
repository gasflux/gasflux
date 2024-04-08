import logging
from datetime import datetime
from pathlib import Path

import numpy as np
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
        self._check_cols()
        self._check_dtypes()
        self._check_ranges()
        logger.info("Data validation passed")

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
                if self.df[col].dtype != "float64":
                    logging.error(f"Column '{col}' is not of type 'float64'.")

    def _check_ranges(self) -> None:
        """Check that the required columns are within the specified ranges."""
        for col, (min_val, max_val) in self.required_cols.items():
            if col in self.df.columns and not self.df[col].between(min_val, max_val, inclusive="both").all():
                logging.error(f"Column '{col}' contains values out of range: {min_val} to {max_val}.")


class BaselineStrategy(ABC):
    def __init__(self, data_processor):
        self.data_processor = data_processor

    @abstractmethod
    def process(self):
        pass


class AlgorithmicBaselineStrategy(BaselineStrategy):
    def process(self):
        logger.info("Applying algorithmic baseline correction")
        for gas in self.data_processor.gases:
            (
                self.data_processor.df,
                self.data_processor.figs["baseline"][gas],
                self.data_processor.text[f"baseline_{gas}"],
            ) = gasflux.baseline.baseline(
                df=self.data_processor.df, gas=gas, algorithm=self.data_processor.config["baseline_algorithm"]
            )
            self.data_processor.std[f"{gas}_baseline"] = np.std(
                self.data_processor.df.loc[~self.data_processor.df[f"{gas}_signal"], f"{gas}_normalised"]
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
        logger.info("Processing InSitu Sensor Data")
        for gas in self.data_processor.gases:
            self.data_processor.figs["scatter_3d"][gas] = gasflux.plotting.scatter_3d(
                df=self.data_processor.df, color=gas, colorbar_title=f"{gas.upper()} flux (kg/m²/h)"
            )
        self.data_processor.figs["windrose"] = gasflux.plotting.windrose(self.data_processor.df, plot_transect=True)
        self.data_processor.figs["wind_timeseries"] = gasflux.plotting.time_series(
            self.data_processor.df, y="windspeed", y2="winddir"
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
                    z=self.data_processor.dfs["removed"]["altitude_ato"],
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
                self.data_processor.krig_parameters[gas],
                self.data_processor.text[f"krig_output_{gas}"],
                self.data_processor.figs["contour"][gas],
                self.data_processor.figs["krig_grid"][gas],
                self.data_processor.figs["semivariogram"][gas],
            ) = gasflux.interpolation.ordinary_kriging(
                df=self.data_processor.df,
                x="x",
                y="altitude_ato",
                gas=gas,
                ordinary_kriging_settings=self.data_processor.config["ordinary_kriging_settings"],
                **self.data_processor.config["variogram_settings"],
            )
            logger.info(f"Kriged {gas}")


class DataProcessor:
    def __init__(self, config, df):
        self.config = config
        self.df = df
        self.gases = config["gases"]
        self.processing_time = datetime.now()
        self.figs: dict = {
            "scatter_3d": {},
            "windrose": None,
            "wind_timeseries": None,
            "baseline": {},
            "contour": {},
            "krig_grid": {},
            "semivariogram": {},
        }
        self.text = {}
        self.std = {}
        self.dfs = {}
        self.reports = {}
        self.krig_parameters = {}
        self.baseline_strategy = AlgorithmicBaselineStrategy(self)
        self.sensor_strategy = InSituSensorStrategy(self)
        self.spatial_processing_strategy = CurtainSpatialProcessingStrategy(self)
        self.interpolation_strategy = KrigingInterpolationStrategy(self)

    def process(self):
        self.df = gasflux.pre_processing.add_utm(self.df)
        self.df = gasflux.pre_processing.add_heading(self.df)
        DataValidator(self.df, self.config).validate()
        self.baseline_strategy.process()
        self.sensor_strategy.process()
        self.spatial_processing_strategy.process()
        self.interpolation_strategy.process()

        # Reporting
        for gas in self.gases:
            self.reports[gas] = gasflux.reporting.mass_balance_report(
                krig_params=self.krig_parameters[gas],
                wind_fig=self.figs["wind_timeseries"],
                baseline_fig=self.figs["baseline"][gas],
                threed_fig=self.figs["scatter_3d"][gas],
                krig_fig=self.figs["contour"][gas],
                windrose_fig=self.figs["windrose"],
            )


def process_main(data_file: Path, config_file: Path | None = None) -> None:
    """Main function to run the pipeline."""
    if config_file is None:
        config_file = Path(__file__).parent / "config.yaml"
    config = load_config(config_file)
    name = data_file.stem
    df = read_csv(data_file)

    processor = DataProcessor(config, df)
    processor.process()

    # write report
    output_dir = Path(config["output_dir"]).expanduser()
    output_path = output_dir / name / processor.processing_time.strftime("%Y-%m-%d_%H-%M-%S-%f_processing_run")
    output_path.mkdir(parents=True, exist_ok=True)
    for gas, report in processor.reports.items():
        with Path.open(output_path / f"{name}_{gas}_report.html", "w") as f:
            f.write(report)
    logger.info(f"Reports written to {output_path}")
