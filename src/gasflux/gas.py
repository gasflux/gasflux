"""functions related to gas transformations and calculations, e.g. density, point flux etc."""

import molmass
import pandas as pd

gas_variables = {
    "standard_pressure": 1013.25,  # hPa/mbar
    "standard_temperature": 273.15,  # degrees K
    "standard_molar_volume": 0.022413969545014,  # m3⋅mol-1
}


def mass(formula: str) -> float:
    """Return the molar mass of a gas in g/mol."""
    return molmass.Formula(formula.upper()).mass  # only accepts capital letters


def gas_density(local_pressure: float, local_temperature_celsius: float, gas: str) -> float:  # millibars and celsius
    """
    Calculate the density of a gas in kg/m3 based on local pressure and temperature.

    Parameters:
    - local_pressure: The local pressure in hPa/mbar.
    - local_temperature: The local temperature in degrees Celsius.
    - gas: The chemical formula of the gas.

    Returns:
    - The density of the gas in kg/m3.

    Assumes ideal gas behavior.
    """
    local_temperature_kelvin = local_temperature_celsius + gas_variables["standard_temperature"]
    local_volume = (
        gas_variables["standard_molar_volume"]
        * (gas_variables["standard_pressure"] / local_pressure)
        * ((local_temperature_kelvin + gas_variables["standard_temperature"]) / gas_variables["standard_temperature"])
    )  # m3⋅mol-1
    return mass(gas) / 1000 / local_volume


def gas_flux_column(df: pd.DataFrame, gas: str, wind: str = "windspeed") -> pd.DataFrame:
    """
    Add columns to the DataFrame for the gas density, mass, and flux.

    Parameters:
    - df: The DataFrame.
    - gas: The chemical formula of the gas.
    - wind: The column name for the wind speed (NB - must be perpendicular to the plane)

    Returns:
    - The DataFrame with the added columns.
    """
    average_temp = df["temperature"].mean()  # celsius
    average_pressure = df["pressure"].mean()  # hPa
    gd = gas_density(local_pressure=average_pressure, local_temperature_celsius=average_temp, gas=gas)  # kg/m3
    df[f"{gas}_kg_m3"] = gd * (df[f"{gas}_normalised"] * 1e-6)  # kg/m3
    df[f"{gas}_kg_h_m2"] = df[f"{gas}_kg_m3"] * df[wind] * 60 * 60  # kg/h/m2
    return df
