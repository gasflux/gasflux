import pandas as pd
import gasflux
import yaml
from pathlib import Path


testdf = pd.read_csv(Path(__file__).parent / "data" / "testdata.csv")
testconfig = yaml.safe_load(open(Path(__file__).parent / "testconfig.yaml"))


def load_cols(cols):
    return testdf[cols]


def test_bimodal_azimuth():
    input_mode = testconfig["transect_azimuth"]
    input_reciprocal_mode = (input_mode + 180) % 360
    df = load_cols(["azimuth_heading", "altitude_ato"])
    mode1, mode2 = gasflux.processing.bimodal_azimuth(df)
    assert (
        mode1 - input_mode < 3 or mode1 - input_reciprocal_mode < 3
    ), "Mode1 does not match expected azimuth or its reciprocal within 3 degrees"

    if mode1 - input_mode < 3:
        assert mode2 - input_reciprocal_mode < 3, "Mode2 does not match expected reciprocal azimuth within 3 degrees"
    else:
        assert mode2 - input_mode < 3, "Mode2 does not match expected azimuth within 3 degrees"


def test_bimodal_elevation():
    df = load_cols(["elevation_heading", "altitude_ato"])
    input_mode = 0
    input_reciprocal_mode = (input_mode + 180) % 360
    mode1, mode2 = gasflux.processing.bimodal_elevation(df)
    assert (
        mode1 - input_mode < 3 or mode1 - input_reciprocal_mode < 3
    ), "Mode1 does not match expected elevation or its reciprocal within 3 degrees"
    if mode1 - input_mode < 3:
        assert mode2 - input_reciprocal_mode < 3, "Mode2 does not match expected reciprocal elevation within 3 degrees"
    else:
        assert mode2 - input_mode < 3, "Mode2 does not match expected elevation within 3 degrees"


def test_altitude_transect_splitter():
    df = load_cols(["altitude_ato"])
    df, fig = gasflux.processing.altitude_transect_splitter(df)
    assert "transect_num" in df.columns, "Transect number column not added to dataframe"
    assert (
        df["transect_num"].nunique() == testconfig["number_of_transects"]
    ), "Dataframe was not split into the right number of transects"
