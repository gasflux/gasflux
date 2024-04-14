import argparse
from pathlib import Path
import shutil

from colorama import init, Fore, Style
from gasflux.processing_pipelines import process_main

init()


def find_config_file(path: Path, recursive: bool = True):
    if recursive:
        config_files = [file for file in path.rglob("*.yaml") if "gasflux_config" in file.read_text()]
    else:
        config_files = [file for file in path.glob("*.yaml") if "gasflux_config" in file.read_text()]
    if len(config_files) == 1:
        return config_files[0]
    elif len(config_files) > 1:
        raise ValueError("Multiple config files found: {}".format(", ".join(str(file) for file in config_files)))
    else:
        raise FileNotFoundError("No config file found in the supplied path or its child folders")


def process_command(data_path: str, config_path: str, test: bool):
    if test:
        data_file = Path(__file__).parent / "testdata" / "testdata.csv"
        config_file = Path(__file__).parent / "testdata" / "testconfig.yaml"
        process_main(data_file, config_file)
    else:
        dpath_obj = Path(data_path)
        if dpath_obj.is_dir():
            data_files = dpath_obj.rglob("*.csv")
            if config_path is None:
                config_file = find_config_file(dpath_obj)
            else:
                config_file = Path(config_path)
            for data_file in data_files:
                process_main(data_file, config_file)
        elif dpath_obj.is_file():
            data_file = dpath_obj
            if config_path is None:
                config_file = find_config_file(dpath_obj.parent, recursive=False)
            else:
                config_file = Path(config_path)
            process_main(data_file, config_file)
        else:
            raise FileNotFoundError(f"Invalid data path: {data_path}")


def generate_config_command(config_destination: str):
    default_config_path = Path(__file__).parent / "gasflux_config.yaml"
    destination_path = Path(config_destination)
    if not destination_path.is_dir():
        raise NotADirectoryError(f"Destination path is not a directory: {config_destination}")
    else:
        destination_path = destination_path / "gasflux_config.yaml"
        shutil.copy(default_config_path, destination_path)
        print(f"Config file copied to: {destination_path}")


def main_cli():
    parser = argparse.ArgumentParser(
        description="GasFlux Processing Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,  # Disable the default help to use the custom help screen
    )
    subparsers = parser.add_subparsers(dest="command")

    process_parser = subparsers.add_parser(
        "process",
        help="Process CSV data files",
        description=(
            "Process CSV data files located in a specified directory or a single file.\n"
            "The program will search for a configuration file (gasflux_config.yaml) in the same directory as the data file(s).\n"  # noqa
            "If no configuration file is found, the default configuration will be used."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    process_parser.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help=(
            "Path to the data file or directory containing CSV files.\n"
            "If a directory is provided, all CSV files within the directory will be processed."
        ),
    )
    process_parser.add_argument(
        "--config-path",
        "-c",
        default="src/gasflux/gasflux_config.yaml",
        help=(
            "Path to the configuration file (gasflux_config.yaml).\n"
            "If not specified, the program will search for the configuration file in the same directory as the data file(s).\n"  # noqa
            "If no configuration file is found, the default configuration will be used."
        ),
    )
    process_parser.add_argument(
        "--test",
        action="store_true",
        help="Use test data instead of the specified data file(s).",
    )

    config_parser = subparsers.add_parser(
        "generate-config",
        help="Generate a default configuration file",
        description=(
            "Generate a default configuration file (gasflux_config.yaml) in the specified directory.\n"
            "If no directory is specified, the configuration file will be generated in the current directory."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    config_parser.add_argument(
        "config_destination",
        nargs="?",
        default=None,
        help="Destination directory for the generated configuration file.",
    )

    # Custom help option
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help message and exit",
    )

    args = parser.parse_args()

    if args.command == "process":
        process_command(args.data_path, args.config_path, args.test)
    elif args.command == "generate-config":
        generate_config_command(args.config_destination)
    elif args.help:
        display_help()
    else:
        display_help()


def display_help():
    help_text = f"""
{Fore.CYAN}GasFlux Processing Pipeline{Style.RESET_ALL}
{Fore.CYAN}==========================={Style.RESET_ALL}

{Fore.YELLOW}Description:{Style.RESET_ALL}
The GasFlux Processing Pipeline is a command-line tool for processing CSV data files and generating configuration files.

{Fore.YELLOW}Usage:{Style.RESET_ALL}
  gasflux [OPTIONS] COMMAND [ARGS]...

{Fore.YELLOW}Commands:{Style.RESET_ALL}
  {Fore.GREEN}process{Style.RESET_ALL}         Process CSV data files
  {Fore.GREEN}generate-config{Style.RESET_ALL} Generate a default configuration file

{Fore.YELLOW}Options:{Style.RESET_ALL}
  {Fore.BLUE}-h, --help{Style.RESET_ALL}      Show this help message and exit
  {Fore.BLUE}-v, --version{Style.RESET_ALL}   Show the version and exit

{Fore.YELLOW}Commands Help:{Style.RESET_ALL}
  {Fore.GREEN}process{Style.RESET_ALL}         Process CSV data files located in a specified directory or a single file.
                  The program will search for a singular configuration file (gasflux_config.yaml) in the same directory
                  - if a directory is supplied to [DATA_PATH] then child directories are searched too.

                  {Fore.YELLOW}Usage:{Style.RESET_ALL} gasflux process [OPTIONS] [DATA_PATH]

                  {Fore.YELLOW}Options:{Style.RESET_ALL}
                    {Fore.BLUE}-c, --config-path PATH{Style.RESET_ALL}  Path to the configuration file (gasflux_config.yaml).
                                            If not specified, the program will search for the configuration file
                                            in the same directory as the data file(s).
                                            If no configuration file is found, the default configuration will be used.
                    {Fore.BLUE}--test{Style.RESET_ALL}                  Use test data instead of the specified data file(s).
                    {Fore.BLUE}-h, --help{Style.RESET_ALL}              Show this help message and exit

  {Fore.GREEN}generate-config{Style.RESET_ALL} Generate a default configuration file (gasflux_config.yaml) in the specified directory.
                  If no directory is specified, the configuration file will be generated in the current directory.

                  {Fore.YELLOW}Usage:{Style.RESET_ALL} gasflux generate-config [OPTIONS] [CONFIG_DESTINATION]

                  {Fore.YELLOW}Options:{Style.RESET_ALL}
                    {Fore.BLUE}-h, --help{Style.RESET_ALL}  Show this help message and exit

{Fore.YELLOW}Examples:{Style.RESET_ALL}
  Process a single CSV file:
    $ gasflux process {Fore.MAGENTA}"/path/to/data.csv"{Style.RESET_ALL}

  Process all CSV files in a directory:
    $ gasflux process {Fore.MAGENTA}"/path/to/data/directory"{Style.RESET_ALL}

  Process a single CSV file with a specific configuration file:
    $ gasflux process {Fore.MAGENTA}"/path/to/data.csv"{Style.RESET_ALL} --config-path {Fore.MAGENTA}/path/to/config.yaml{Style.RESET_ALL}

  Process test data:
    $ gasflux process {Fore.BLUE}--test{Style.RESET_ALL}

  Generate a default configuration file in the current directory:
    $ gasflux generate-config {Fore.MAGENTA}.{Style.RESET_ALL}

  Generate a default configuration file in a specific directory:
    $ gasflux generate-config {Fore.MAGENTA}"/path/to/directory"{Style.RESET_ALL}
"""  # noqa
    print(help_text)


if __name__ == "__main__":
    main_cli()
