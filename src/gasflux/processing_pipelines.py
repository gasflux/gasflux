"""this module contains the processing pipelines for the gasflux data"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

import gasflux

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_csv(file_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {e}")
        raise


def load_config(config_path: Path) -> Dict:
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise


class DataValidator:  # TODO decide whether to move this to preprocessing
    def __init__(self, df: pd.DataFrame, config: Dict):
        self.df = df
        self.required_cols = config['required_cols'].copy()
        self.required_cols.update(config['gases'])

    def validate(self) -> None:
        self._check_cols()
        self._check_dtypes()
        self._check_ranges()
        logger.info("Data validation passed")

    def _check_cols(self) -> None:
        missing_cols = set(self.required_cols) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing or mislabelled columns: {missing_cols}")

    def _check_dtypes(self) -> None:
        for col in self.required_cols:
            if col in self.df:
                if self.df[col].isna().any():
                    raise ValueError(f"Column '{col}' contains NaN values.")
                if self.df[col].dtype != "float64":
                    raise TypeError(f"Column '{col}' is not of type 'float64'.")

    def _check_ranges(self) -> None:
        for col, (min_val, max_val) in self.required_cols.items():
            if col in self.df and not self.df[col].between(min_val, max_val, inclusive="both").all():
                raise ValueError(f"Column '{col}' contains values out of range: {min_val} to {max_val}.")


class CurtainPipeline:
    def __init__(self, df: pd.DataFrame, config: Dict, name: str):
        self.processing_time = datetime.now()
        self.dfs: Dict = {}
        self.figs: Dict = {"baseline": {},
                           "scatter_3d": {},
                           "windrose": go.Figure,
                           "wind_timeseries": go.Figure,
                           "contour": {},
                           "krig_grid": {},
                           "semivariogram": {}
                           }
        self.text: Dict = {"krig_output": {}}
        self.std: Dict = {}
        self.reports: Dict = {}
        self.krig_parameters: Dict = {}
        self.config = config
        self.df = df
        self.gases = config['gases']
        self.name = name

    def run(self) -> None:
        validator = DataValidator(self.df, self.config)
        validator.validate()
        # Add columns
        self.df = gasflux.pre_processing.add_utm(self.df)
        self.df = gasflux.pre_processing.add_heading(self.df)
        # Baseline correction
        for gas in self.gases:
            self.df, self.figs["baseline"][gas], self.text[f"baseline_{gas}"] = gasflux.baseline.baseline(df=self.df, y=gas, algorithm=self.config["baseline_algorithm"])
        self.std[f"baseline_{gas}"] = np.std(self.df.loc[~self.df["signal"], f"{gas}_normalised"])
        self.df_std = self.df.copy()
        # Filtering of flight
        self.dfs["original"] = self.df.copy()
        self.df, self.start_transect, self.end_transect = gasflux.processing.largest_monotonic_transect_series(self.df)
        # TODO - add filtering
        self.dfs["removed"] = self.dfs["original"].loc[self.dfs["original"].index.difference(self.df.index)]

        # Orthogonal distance regression to flatten the plane
        self.df, self.plane_angle = gasflux.processing.flatten_linear_plane(self.df)
        self.df = gasflux.gas.methane_flux_column(self.df)

        # Graphs
        for gas in self.gases:
            self.figs["scatter_3d"][gas] = gasflux.plotting.scatter_3d(self.df, gas)
            self.figs["scatter_3d"][gas].add_trace(  # TODO - put this in plotting
                go.Scatter3d(
                    x=self.dfs["removed"]["utm_easting"],
                    y=self.dfs["removed"]["utm_northing"],
                    z=self.dfs["removed"]["altitude_ato"],
                    mode="markers",
                    marker=dict(size=2, color="black", symbol="circle", opacity=0.5),
                ))
        self.figs["windrose"] = gasflux.plotting.windrose(self.df, plot_transect=True)
        self.figs["wind_timeseries"] = gasflux.plotting.time_series(self.df, y="windspeed", y2="winddir")

        # Interpolation
        for gas in self.gases:
            self.krig_parameters[gas], self.text["krig_output"][gas], self.figs["contour"][gas], self.figs["krig_grid"][gas], self.figs["semivariogram"][gas] = gasflux.interpolation.ordinary_kriging(
                df=self.df,
                x='x',
                y="altitude_ato",
                gas="ch4_kg_h_m2",
                ordinary_kriging_settings=self.config['ordinary_kriging_settings'],
                **self.config['variogram_settings'])
            print(f"Kriged {self.name} {gas}")

        # Reporting
        for gas in self.gases:
            try:
                self.reports[gas] = gasflux.reporting.mass_balance_report(df=self.df,
                                                                          krig_params=self.krig_parameters[gas],
                                                                          wind_fig=self.figs["wind_timeseries"],
                                                                          baseline_fig=self.figs["baseline"][gas],
                                                                          threed_fig=self.figs["scatter_3d"][gas],
                                                                          krig_fig=self.figs["contour"][gas],
                                                                          windrose_fig=self.figs["windrose"])
            except KeyError as e:
                logger.error(f"Missing data or figures for {gas}: {e}")


def main():
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)
    data_file = Path(__file__).parent.parent.parent / "tests" / "data" / "test_data.csv"
    name = data_file.stem
    df = read_csv(data_file)
    curtain = CurtainPipeline(df, config, name)
    curtain.run()

    # write report
    output_dir = Path(config["output_dir"]).expanduser()
    output_path = output_dir / name / curtain.processing_time.strftime("%Y-%m-%d_%H-%M-%S_%f_processing_run")
    output_path.mkdir(parents=True, exist_ok=True)
    for gas, report in curtain.reports.items():
        with open(output_path / f"{gas}_report.html", "w") as f:
            f.write(report)
    print("done")


if __name__ == "__main__":
    main()
