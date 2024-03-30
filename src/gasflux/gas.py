"""functions related to gas transformations and calculations, e.g. density, point flux etc."""

import pandas as pd

gas_variables = {
    "standard_pressure": 1013.25,  # mbar
    "standard_temperature": 273.15,  # K
    "methane_molar_mass": 16.04,  # g⋅mol−1
    "standard_methane_molar_volume": 0.022413,  # m3⋅mol−1
}


# def gas_local_volume(df: pd.DataFrame, local_pressure, local_temperature) -> float:
#     return (  # type: ignore
#         standard_molar_volume
#         * (standard_pressure / local_pressure)
#         * ((local_temperature + standard_temperature) / standard_temperature)
#     )  # m3⋅mol−1


# def ch4_density(df: pd.DataFrame) -> float:
#     return CH4_molar_mass / 1000 / gas_local_volume(df)  # kg⋅m-3


def ch4_density(df: pd.DataFrame, local_pressure, local_temperature) -> float:  # millibars and celsius
    methane_local_volume = (
        gas_variables["standard_methane_molar_volume"]
        * (gas_variables["standard_pressure"] / local_pressure)
        * ((local_temperature + gas_variables["standard_temperature"]) / gas_variables["standard_temperature"])
    )  # m3⋅mol−1
    return gas_variables["methane_molar_mass"] / 1000 / methane_local_volume  # kg⋅m-3


def methane_flux_column(
    df: pd.DataFrame,
    gas: str = "ch4",
    wind: str = "windspeed",
) -> pd.DataFrame:
    average_temp = df["temperature"].mean()  # celsius
    average_pressure = df["pressure"].mean()  # hPa
    methane_density = ch4_density(df, local_pressure=average_pressure, local_temperature=average_temp)  # kg/m3
    df["ch4_kg_m3"] = methane_density * (df[f"{gas}_normalised"] * 1e-6)  # kg/m3
    df["ch4_kg_h_m2"] = df["ch4_kg_m3"] * df[wind] * 60 * 60  # kg/h/m2
    return df
