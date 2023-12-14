import joblib
import pandas as pd
import plotting

# Load the model
model = joblib.load('model.pkl')


def make_prediction(df: pd.DataFrame, azimuth_heading='azimuth_heading', elevation_heading='elevation_heading',
                    altitude_ato='altitude_ato', idx='idx', percent_complete='percent_complete',
                    horiz_spd='airdata.horiz_spd', z_spd='airdata.z_spd', longitude='airdata.longitude',
                    latitude='airdata.latitude', flight='flight'):
    """
    Make predictions based on the input DataFrame and add them to the DataFrame.
    :param df: DataFrame containing the input data.
    :param azimuth_heading: Column name for azimuth_heading.
    :param elevation_heading: Column name for elevation_heading.
    :param altitude_ato: Column name for altitude_ato.
    :param idx: Column name for idx.
    :param percent_complete: Column name for percent_complete.
    :param horiz_spd: Column name for horizontal speed.
    :param z_spd: Column name for z speed.
    :param longitude: Column name for longitude.
    :param latitude: Column name for latitude.
    :param flight: Column name for flight.
    :return: DataFrame with added predictions column.
    """

    # Ensure the DataFrame contains all the required features
    required_features = [azimuth_heading, elevation_heading, altitude_ato, idx,
                         percent_complete, horiz_spd, z_spd]
    if not all(feature in df.columns for feature in required_features):
        raise ValueError("DataFrame is missing one or more required features.")

    # Make predictions using the specified columns
    predictions = model.predict(df[required_features])

    # Add predictions to the DataFrame
    df['predictions'] = predictions
    fig = plotting.scatter_3d_filter(df)

    return df, fig
