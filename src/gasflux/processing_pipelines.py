"""Processing pipelines for the gasflux data."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

import gasflux

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
            raise

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


class CurtainPipeline:
    """Processing pipeline for the curtain data (2D on a cross-sectional plane)."""

    def __init__(self, df: pd.DataFrame, config: dict, name: str):
        """Initialise the pipeline with the DataFrame, config, and name."""
        self.processing_time = datetime.now()
        self.dfs: dict = {}
        self.figs: dict = {"baseline": {},
                           "scatter_3d": {},
                           "windrose": go.Figure,
                           "wind_timeseries": go.Figure,
                           "contour": {},
                           "krig_grid": {},
                           "semivariogram": {},
                           }
        self.text: dict = {"krig_output": {}}
        self.std: dict = {}
        self.reports: dict = {}
        self.krig_parameters: dict = {}
        self.config = config
        self.df = df
        self.gases = config["gases"]
        self.name = name

    def run(self) -> None:
        """Run the processing pipeline."""
        validator = DataValidator(self.df, self.config)
        validator.validate()
        # Add columns
        self.df = gasflux.pre_processing.add_utm(self.df)
        self.df = gasflux.pre_processing.add_heading(self.df)
        # Baseline correction
        for gas in self.gases:
            self.df, self.figs["baseline"][gas], self.text[f"baseline_{gas}"] = \
                gasflux.baseline.baseline(df=self.df, gas=gas, algorithm=self.config["baseline_algorithm"])
            self.std[f"{gas}_baseline"] = np.std(self.df.loc[~self.df[f"{gas}_signal"], f"{gas}_normalised"])
        self.df_std = self.df.copy()
        # Filtering of flight
        self.dfs["original"] = self.df.copy()
        self.df, self.start_transect, self.end_transect = gasflux.processing.largest_monotonic_transect_series(self.df)
        # TODO - add filtering
        self.dfs["removed"] = self.dfs["original"].loc[self.dfs["original"].index.difference(self.df.index)]

        # Orthogonal distance regression to flatten the plane
        self.df, self.plane_angle = gasflux.processing.flatten_linear_plane(self.df)

        # Add flux column based on wind speed and direction
        for gas in self.gases:
            self.df = gasflux.gas.gas_flux_column(self.df, gas)

        # Graphs
        for gas in self.gases:
            self.figs["scatter_3d"][gas] = gasflux.plotting.scatter_3d(df=self.df,
                                                                       color=gas,
                                                                       colorbar_title=f"{gas.upper()} flux (kg/m²/h)")
            self.figs["scatter_3d"][gas].add_trace(  # TODO - put this in plotting
                go.Scatter3d(
                    x=self.dfs["removed"]["utm_easting"],
                    y=self.dfs["removed"]["utm_northing"],
                    z=self.dfs["removed"]["altitude_ato"],
                    mode="markers",
                    marker={"size": 2, "color": "black", "symbol": "circle", "opacity": 0.5},
                ))
        self.figs["windrose"] = gasflux.plotting.windrose(self.df, plot_transect=True)
        self.figs["wind_timeseries"] = gasflux.plotting.time_series(self.df, y="windspeed", y2="winddir")

        # Interpolation
        for gas in self.gases:
            self.krig_parameters[gas], self.text["krig_output"][gas], self.figs["contour"][gas], \
                self.figs["krig_grid"][gas], self.figs["semivariogram"][gas] \
                = gasflux.interpolation.ordinary_kriging(
                    df=self.df,
                    x="x",
                    y="altitude_ato",
                    gas="ch4_kg_h_m2",
                    ordinary_kriging_settings=self.config["ordinary_kriging_settings"],
                    **self.config["variogram_settings"])
            logger.info(f"Kriged {self.name} {gas}")

        # Reporting
        self.figs["scatter_3d"]["multigas"] = gasflux.plotting.scatter_3d_multigas(df=self.df, gases=self.gases)
        self.figs["baseline"]["multigas"] = gasflux.plotting.baseline_plotting_multigas(self.figs["baseline"])
        self.figs["contour"]["multigas"] = gasflux.plotting.contour_krig_multigas(self.figs["contour"])
        # TODO add other figs

        self.reports[gas] = gasflux.reporting.mass_balance_report(krig_params=self.krig_parameters["ch4"],  # TODO - fix
                                                                  wind_fig=self.figs["wind_timeseries"],
                                                                  baseline_fig=self.figs["baseline"]["multigas"],
                                                                  threed_fig=self.figs["scatter_3d"]["multigas"],
                                                                  krig_fig=self.figs["contour"]["multigas"],
                                                                  windrose_fig=self.figs["windrose"])


def main() -> None:
    """Main function to run the pipeline."""
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)
    data_file = Path(__file__).parent.parent.parent / "tests" / "data" / "testdata.csv"
    name = data_file.stem
    df = read_csv(data_file)
    curtain = CurtainPipeline(df, config, name)
    curtain.run()

    # write report
    output_dir = Path(config["output_dir"]).expanduser()
    output_path = output_dir / name / curtain.processing_time.strftime("%Y-%m-%d_%H-%M-%S-%f_processing_run")
    output_path.mkdir(parents=True, exist_ok=True)
    for gas, report in curtain.reports.items():
        with Path.open(output_path / f"{name}_{gas}_report.html", "w") as f:
            f.write(report)


if __name__ == "__main__":
    main()
