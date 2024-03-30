import os

import joblib
import pandas as pd

from . import plotting

model = None
# Lazy loading: Load the model only if it hasn't been loaded yet


def load_model():
    global model
    if model is None:
        default_model_path = os.path.join(os.path.dirname(__file__), 'resources/model.pkl')
        model_file_path = os.getenv("GASFLUX_MODEL_PATH", default_model_path)
        try:
            model = joblib.load(model_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_file_path}. Please check the file path.")
        except Exception as e:
            raise Exception(f"An error occurred while loading the model: {str(e)}")

    return model


def make_prediction(df: pd.DataFrame, elevation_heading='elevation_heading',
                    altitude_ato='altitude_ato',
                    horiz_spd='horiz_spd', z_spd='z_spd'):
    """
    Make predictions based on the input DataFrame and add them to the DataFrame.
    :param df: DataFrame containing the required features
    :param azimuth_heading: Name of the column containing the azimuth heading
    :param elevation_heading: Name of the column containing the elevation heading
    :param altitude_ato: Name of the column containing the altitude above take-off
    :param idx: Name of the column containing the index
    """
    model = load_model()
    # Ensure the DataFrame contains all the required features
    required_features = [elevation_heading, altitude_ato, horiz_spd, z_spd]
    if not all(feature in df.columns for feature in required_features):
        missing_features = [feature for feature in required_features if feature not in df.columns]
        raise ValueError(f"DataFrame is missing (or mislabelled) the following required features: {missing_features}")
    # make idx col if not present
    if 'idx' not in df.columns:
        df['idx'] = df.index
    cols_for_model = ['elevation_heading', 'altitude_ato', 'idx', 'horiz_spd', 'z_spd']
    predictions = model.predict(df[cols_for_model])

    df['predictions'] = predictions
    fig = plotting.scatter_3d(df)

    return df, fig
