import argparse
from pathlib import Path
import shutil

from gasflux.processing_pipelines import process_main


def process_command(data_path: str, config_path: str, test: bool):
    config_file = Path(config_path) if config_path else None
    if test:
        data_file = Path(__file__).parent.parent.parent / "tests" / "data" / "testdata.csv"
        process_main(data_file, config_file)
    else:
        if Path(data_path).is_dir():
            data_files = Path(data_path).rglob("*.csv")
            for data_file in data_files:
                process_main(data_file, config_file)
        if Path(data_path).is_file():
            data_file = Path(data_path)
            process_main(data_file, config_file)


def generate_config_command(config_destination: str | None = None):
    default_config_path = Path(__file__).parent / "config.yaml"
    destination_path = Path(config_destination) if config_destination else Path.cwd()

    if destination_path.is_dir():
        destination_path = destination_path / "config.yaml"

    shutil.copy(default_config_path, destination_path)
    print(f"Config file copied to: {destination_path}")


def main_cli():
    parser = argparse.ArgumentParser(description="GasFlux Processing Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    process_parser = subparsers.add_parser(
        "process", help="Process csv data in a supplied path, or all csv files in a supplied directory"
    )
    process_parser.add_argument("data_path", nargs="?", default=None, help="Path to the data file")
    process_parser.add_argument(
        "--config", "-c", default="src/gasflux/config.yaml", help="Path to the configuration file"
    )
    process_parser.add_argument("--test", action="store_true", help="Use test data")

    config_parser = subparsers.add_parser(
        "generate-config", help="Generate a config file in the current directory or a supplied path"
    )
    config_parser.add_argument(
        "config_destination", nargs="?", default=None, help="Destination path for the config file"
    )

    args = parser.parse_args()

    if args.command == "process":
        process_command(args.data_path, args.config, args.test)
    elif args.command == "generate-config":
        generate_config_command(args.config_destination)
    else:
        parser.print_help()


if __name__ == "__main__":
    main_cli()


if __name__ == "__main__":
    main_cli()
