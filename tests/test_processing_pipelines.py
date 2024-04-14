import gasflux
import yaml
from pathlib import Path
import pytest


@pytest.fixture
def setup_test_environment(tmp_path):
    """Prepare actual test data and configuration with a modified output directory."""
    df_path = Path(__file__).parents[1] / "src" / "gasflux" / "testdata" / "testdata.csv"
    config_path = Path(__file__).parents[1] / "src" / "gasflux" / "testdata" / "testconfig.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["output_dir"] = str(tmp_path)

    temp_config_path = tmp_path / "temp_testconfig.yaml"
    with open(temp_config_path, "w") as f:
        yaml.safe_dump(config, f)

    return df_path, temp_config_path


def test_process_main_config_output(setup_test_environment):
    df_path, temp_config_path = setup_test_environment
    gasflux.processing_pipelines.process_main(df_path, temp_config_path)
    with open(temp_config_path) as f:
        temp_config = yaml.safe_load(f)
    output_dir = Path(temp_config["output_dir"]) / df_path.stem
    assert output_dir.exists(), "Output directory does not exist."
    processing_run_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    assert len(processing_run_dirs) > 0, "No processing run directory found."
    processing_run_dir = processing_run_dirs[0]

    with open(temp_config_path) as f:
        original_config = yaml.safe_load(f)

    for gas in original_config.get("gases", []):
        report_path = processing_run_dir / f"{df_path.stem}_{gas}_report.html"
        assert report_path.exists(), f"Report for {gas} does not exist."

    config_dump_path = processing_run_dir / f"{df_path.stem}_config.yaml"
    assert config_dump_path.exists(), "Config dump file does not exist."

    def load_and_redump(yaml_file):
        with open(yaml_file) as file:
            data = yaml.safe_load(file)
            redumped_data = yaml.dump(data, sort_keys=True, default_flow_style=False)
        return redumped_data

    assert load_and_redump(temp_config_path) == load_and_redump(
        config_dump_path
    ), "The dumped config does not match the input config."


def test_process_main_deterministic_output(setup_test_environment):
    df_path, temp_config_path = setup_test_environment
    gasflux.processing_pipelines.process_main(df_path, temp_config_path)
