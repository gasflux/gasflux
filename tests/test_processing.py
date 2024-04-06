import pandas as pd
import gasflux
import yaml
from pathlib import Path
import numpy as np
from gasflux.processing import circ_dist


testdf = pd.read_csv(Path(__file__).parent / "data" / "testdata.csv")
testconfig = yaml.safe_load(open(Path(__file__).parent / "testconfig.yaml"))


def load_cols(cols):
    return testdf[cols]


def test_min_angular_diff_def():
    a = 0
    b = 359
    diff = gasflux.processing.circ_dist(a, b)
    assert diff == 1, "Angular difference not calculated correctly"


def test_circ_median():
    x = np.array([0, 1, 2, 359, 4, 3])
    median = gasflux.processing.circ_median(x)
    assert median == 1.5, "Circular median not calculated correctly"


def test_bimodal_azimuth():
    input_mode = testconfig["transect_azimuth"]
    input_reciprocal_mode = (input_mode + 180) % 360
    df = load_cols(["azimuth_heading", "altitude_ato"])
    mode1, mode2 = gasflux.processing.bimodal_azimuth(df)
    assert (
        circ_dist(mode1, input_mode) < 3 or circ_dist(mode1, input_reciprocal_mode) < 3
    ), "Mode1 does not match expected azimuth or its reciprocal within 3 degrees"

    if circ_dist(mode1, input_mode) < 3:
        assert (
            circ_dist(mode2, input_reciprocal_mode) < 3
        ), "Mode2 does not match expected reciprocal azimuth within 3 degrees"
    else:
        assert circ_dist(mode2, input_mode) < 3, "Mode2 does not match expected azimuth within 3 degrees"


def test_bimodal_elevation():
    df = load_cols(["elevation_heading", "altitude_ato"])
    input_mode = 0
    input_reciprocal_mode = 0 - input_mode
    mode1, mode2 = gasflux.processing.bimodal_elevation(df)
    assert (
        circ_dist(mode1, input_mode) < 3 or circ_dist(mode1, input_reciprocal_mode) < 3
    ), "Mode1 does not match expected elevation or its reciprocal within 3 degrees"
    if circ_dist(mode1, input_mode) < 3:
        assert (
            circ_dist(mode2, input_reciprocal_mode) < 3
        ), "Mode2 does not match expected reciprocal elevation within 3 degrees"
    else:
        assert circ_dist(mode2, input_mode) < 3, "Mode2 does not match expected elevation within 3 degrees"


def test_altitude_transect_splitter():
    df = load_cols(["altitude_ato"])
    df, fig = gasflux.processing.altitude_transect_splitter(df)
    assert "transect_num" in df.columns, "Transect number column not added to dataframe"
    assert (
        df["transect_num"].nunique() == testconfig["number_of_transects"]
    ), "Dataframe was not split into the right number of transects"


def test_add_transect_azimuth_switches():
    df = load_cols(["azimuth_heading"])
    df = gasflux.processing.add_transect_azimuth_switches(df)
    assert (
        df["transect_num"].nunique() == testconfig["number_of_transects"]
    ), "Transect azimuth switches not added to dataframe"


def test_heading_filter():
    df = load_cols(["azimuth_heading", "elevation_heading", "altitude_ato"])
    azimuth_filter = testconfig["filters"]["heading_filter"]["azimuth_filter"]
    azimuth_window = testconfig["filters"]["heading_filter"]["azimuth_window"]
    elevation_filter = testconfig["filters"]["heading_filter"]["elevation_filter"]
    df_filtered, df_unfiltered = gasflux.processing.heading_filter(
        df, azimuth_filter=azimuth_filter, azimuth_window=azimuth_window, elevation_filter=elevation_filter
    )
    input_mode = testconfig["transect_azimuth"]
    input_reciprocal_mode = (input_mode + 180) % 360
    # assert that the filtered dataframe contains the expected azimuth or its reciprocal within the window
    df_filtered["near_mode1"] = df_filtered["rolling_azimuth_heading"].apply(
        lambda x: circ_dist(x, input_mode) < azimuth_window
    )
    df_filtered["near_mode2"] = df_filtered["rolling_azimuth_heading"].apply(
        lambda x: circ_dist(x, input_reciprocal_mode) < azimuth_window
    )
    assert (
        df_filtered["near_mode1"].any() or df_filtered["near_mode2"].any()
    ), "Filtered dataframe does not contain expected azimuth or its reciprocal within the window"
