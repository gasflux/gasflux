"""Experimental module for machine learning flight filtering"""

import os

import joblib
import pandas as pd

from . import plotting
import plotly.graph_objects as go

model = None  # Lazy loading: Load the model only if it hasn't been loaded yet


def load_model():
    """Load the model from the model file path. If the model has already been loaded, return it."""
    global model
    if model is None:
        default_model_path = os.path.join(os.path.dirname(__file__), "resources/model.pkl")
        model_file_path = os.getenv("GASFLUX_MODEL_PATH", default_model_path)
        try:
            model = joblib.load(model_file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found at {model_file_path}. Please check the file path.") from e
        except Exception as e:
            raise Exception("An error occurred while loading the model.") from e

    return model


def make_prediction(
    df: pd.DataFrame,
    course_elevation="course_elevation",
    height_ato="height_ato",
    horiz_spd="horiz_spd",
    z_spd="z_spd",
) -> tuple[pd.DataFrame, go.Figure]:
    """Make predictions based on the input DataFrame and add them to the DataFrame.
    :param df: DataFrame containing the required features
    :param course_azimuth: Name of the column containing the course azimuth
    :param course_elevation: Name of the column containing the course elevation
    :param height_ato: Name of the column containing the height above take-off
    :param idx: Name of the column containing the index

    return: Tuple of the DataFrame with the predictions and a Plotly 3D scatter plot of the predictions
    """
    model = load_model()
    # Ensure the DataFrame contains all the required features
    required_features = [course_elevation, height_ato, horiz_spd, z_spd]
    if not all(feature in df.columns for feature in required_features):
        missing_features = [feature for feature in required_features if feature not in df.columns]
        raise ValueError(f"DataFrame is missing (or mislabelled) the following required features: {missing_features}")
    # make idx col if not present
    if "idx" not in df.columns:
        df["idx"] = df.index
    cols_for_model = ["course_elevation", "height_ato", "idx", "horiz_spd", "z_spd"]
    predictions = model.predict(df[cols_for_model])

    df["predictions"] = predictions
    fig = plotting.scatter_3d(df)

    return df, fig
