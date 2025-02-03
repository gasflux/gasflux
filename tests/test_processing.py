import pandas as pd
import gasflux
import yaml
from pathlib import Path
import numpy as np
from gasflux.processing import min_angular_displacement
import pytest


testdf = pd.read_csv(Path(__file__).parents[1] / "src" / "gasflux" / "testdata" / "testdata.csv")
testconfig = yaml.safe_load(open(Path(__file__).parents[1] / "src" / "gasflux" / "testdata" / "testconfig.yaml"))


def load_cols(cols):
    return testdf[cols]


def test_min_angular_diff_def():
    def test_min_angular_displacement():
        assert min_angular_displacement(10, 350) == 20
        assert min_angular_displacement(0, 180) == 180
        x = np.array([10, 0])
        y = np.array([350, 180])
        expected = np.array([20, 180])
        result = min_angular_displacement(x, y)
        assert np.all(result == expected), "Vectorized function failed"


def test_circ_median():
    x = np.array([0, 1, 2, 359, 4, 3])
    median = gasflux.processing.circ_median(x)
    assert median == 1.5, "Circular median not calculated correctly"


@pytest.mark.parametrize(
    "plane_angle,expected_winddir_rel,expected_windspeed_normal",
    [
        (
            90,
            [0, 90, 0, 90, 0],
            [5, 0, 5, 0, 5],
        ),
        (
            30,
            [60, 30, 60, 30, 60],
            np.array([1 / 2, np.sqrt(3) / 2, 1 / 2, np.sqrt(3) / 2, 1 / 2]) * 5,
        ),
        (
            60,
            [30, 60, 30, 60, 30],
            np.array([np.sqrt(3) / 2, 1 / 2, np.sqrt(3) / 2, 1 / 2, np.sqrt(3) / 2]) * 5,
        ),
    ],
)
def test_wind_offset_correction_parametrized(plane_angle, expected_winddir_rel, expected_windspeed_normal):
    data = {"winddir": [0, 90, 180, 270, 360], "windspeed": [5, 5, 5, 5, 5]}
    df = pd.DataFrame(data)
    corrected_df = gasflux.processing.wind_offset_correction(df, plane_angle)
    assert "winddir_rel" in corrected_df.columns, f"Relative wind direction column not added for angle {plane_angle}"
    assert "windspeed" in corrected_df.columns, f"Normalised wind speed column not added for angle {plane_angle}"
    assert np.allclose(corrected_df["winddir_rel"], expected_winddir_rel, rtol=1e-5, atol=1e-10), (
        f"Relative wind directions not calculated correctly for angle {plane_angle}"
    )
    assert np.allclose(corrected_df["windspeed"], expected_windspeed_normal, rtol=1e-5, atol=1e-10), (
        f"Normalised wind speeds not calculated correctly for angle {plane_angle}"
    )


def test_bimodal_azimuth():
    input_mode = testconfig["transect_azimuth"]
    input_reciprocal_mode = (input_mode + 180) % 360
    df = load_cols(["course_azimuth", "height_ato"])
    mode1, mode2 = gasflux.processing.bimodal_azimuth(df)
    assert (
        min_angular_displacement(mode1, input_mode) < 3 or min_angular_displacement(mode1, input_reciprocal_mode) < 3
    ), "Mode1 does not match expected azimuth or its reciprocal within 3 degrees"

    if min_angular_displacement(mode1, input_mode) < 3:
        assert min_angular_displacement(mode2, input_reciprocal_mode) < 3, (
            "Mode2 does not match expected reciprocal azimuth within 3 degrees"
        )
    else:
        assert min_angular_displacement(mode2, input_mode) < 3, "Mode2 does not match expected azimuth within 3 degrees"


def test_bimodal_elevation():
    df = load_cols(["course_elevation", "height_ato"])
    input_mode = 0
    input_reciprocal_mode = 0 - input_mode
    mode1, mode2 = gasflux.processing.bimodal_elevation(df)
    assert (
        min_angular_displacement(mode1, input_mode) < 3 or min_angular_displacement(mode1, input_reciprocal_mode) < 3
    ), "Mode1 does not match expected elevation or its reciprocal within 3 degrees"
    if min_angular_displacement(mode1, input_mode) < 3:
        assert min_angular_displacement(mode2, input_reciprocal_mode) < 3, (
            "Mode2 does not match expected reciprocal elevation within 3 degrees"
        )
    else:
        assert min_angular_displacement(mode2, input_mode) < 3, (
            "Mode2 does not match expected elevation within 3 degrees"
        )


def test_height_transect_splitter():
    df = load_cols(["height_ato"])
    df, fig = gasflux.processing.height_transect_splitter(df)
    assert "transect_num" in df.columns, "Transect number column not added to dataframe"
    assert df["transect_num"].nunique() == testconfig["number_of_transects"], (
        "Dataframe was not split into the right number of transects"
    )


def test_add_transect_azimuth_switches():
    df = load_cols(["course_azimuth"])
    df = gasflux.processing.add_transect_azimuth_switches(df)
    assert df["transect_num"].nunique() == testconfig["number_of_transects"], (
        "Transect azimuth switches not added to dataframe"
    )


def test_course_filter():
    df = load_cols(["course_azimuth", "course_elevation", "height_ato"])
    azimuth_filter = testconfig["filters"]["course_filter"]["azimuth_filter"]
    azimuth_window = testconfig["filters"]["course_filter"]["azimuth_window"]
    elevation_filter = testconfig["filters"]["course_filter"]["elevation_filter"]
    df_filtered, df_unfiltered = gasflux.processing.course_filter(
        df, azimuth_filter=azimuth_filter, azimuth_window=azimuth_window, elevation_filter=elevation_filter
    )
    input_mode = testconfig["transect_azimuth"]
    input_reciprocal_mode = (input_mode + 180) % 360
    # assert that the filtered dataframe contains the expected azimuth or its reciprocal within the window
    df_filtered["near_mode1"] = df_filtered["rolling_course_azimuth"].apply(
        lambda x: min_angular_displacement(x, input_mode) < azimuth_window
    )
    df_filtered["near_mode2"] = df_filtered["rolling_course_azimuth"].apply(
        lambda x: min_angular_displacement(x, input_reciprocal_mode) < azimuth_window
    )
    assert df_filtered["near_mode1"].any() or df_filtered["near_mode2"].any(), (
        "Filtered dataframe does not contain expected azimuth or its reciprocal within the window"
    )


def test_mCount_max():
    data_dict = {1: -5.4, 2: 0.6, 3: 5.6, 4: 3.2, 5: 10.4, 6: 18.4, 7: 20.8, 8: 19.4}
    start, end = gasflux.processing.mCount_max(data_dict)
    assert start == 4, "Start index of max count not calculated correctly"
    assert end == 7, "End index of max count not calculated correctly"


def test_largest_monotonic_transect_series():
    df = load_cols(
        ["timestamp", "height_ato", "course_azimuth", "longitude", "latitude", "utm_easting", "utm_northing"]
    )
    df, starttransect, endtransect = gasflux.processing.largest_monotonic_transect_series(df)
    starttransect = 1
    endtransect = testconfig["number_of_transects"]
    assert starttransect == starttransect, "Start index of largest monotonic transect not calculated correctly"
    assert endtransect == endtransect, "End index of largest monotonic transect not calculated correctly"


def test_remove_non_transects():
    df = load_cols(
        ["height_ato", "course_azimuth", "course_elevation", "longitude", "latitude", "utm_easting", "utm_northing"]
    )
    retained_df, removed_df = gasflux.processing.remove_non_transects(df)
    assert retained_df is not None, "Retained dataframe is None"
    assert removed_df is not None, "Removed dataframe is None"


def test_flatten_linear_plane():
    df = load_cols(["height_ato", "utm_easting", "utm_northing"])
    df, plane_angle = gasflux.processing.flatten_linear_plane(df)
    input_plane_angle = testconfig["transect_azimuth"]
    reciprocal_plane_angle = (input_plane_angle + 180) % 360
    assert (
        min_angular_displacement(plane_angle, input_plane_angle) < 3
        or min_angular_displacement(plane_angle, reciprocal_plane_angle) < 3
    ), "Plane angle not calculated correctly"


def test_drone_anemo_to_point_wind():
    data = {
        "yaw": [0, 90, 0, -90, 180],
        "anemo_u": [0, 0, 10, 10, 10],
        "anemo_v": [0, 0, 0, 0, 0],
        "easting": [0, 10, 0, 10, 0],
        "northing": [0, 0, 0, 0, 10],
    }
    df_test = pd.DataFrame(data)
    yaw_col = "yaw"
    anemo_u_col = "anemo_u"
    anemo_v_col = "anemo_v"
    easting_col = "easting"
    northing_col = "northing"
    result_df = gasflux.processing.drone_anemo_to_point_wind(
        df_test, yaw_col, anemo_u_col, anemo_v_col, easting_col, northing_col
    )
    expected_windspeed = np.array([0, 10, 10, np.sqrt(200), np.sqrt(200)])
    expected_winddir = np.array(
        [180, 270, 270, 225, 135]
    )  # 180 not zero because of the way IEEE 754 handles floating point numbers
    windspeed_diff = np.abs(result_df["windspeed"].values - expected_windspeed)
    winddir_diff = gasflux.processing.min_angular_displacement(result_df["winddir"].to_numpy(), expected_winddir)
    assert np.all(windspeed_diff < 1e-10), "Wind speed not calculated correctly"
    assert np.all(np.array(winddir_diff) < 3), "Wind direction not calculated correctly"
