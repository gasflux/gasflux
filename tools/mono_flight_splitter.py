"""a script to split a folder of flights into multiple flights based on the monotonic changes in altitude.
Takes a csv file as an input and will split it out into several csvs.
Needs azimuth headings to work properly.
"""

import argparse
from pathlib import Path

import colorama
import numpy as np
import pandas as pd

import gasflux

colorama.init()


def main(target_dir, search_string, output_dir=None):
    df_list = {}
    for file in Path(target_dir).rglob(search_string):
        df = pd.read_csv(file)
        df_list[file] = df

    for path, df in df_list.items():
        print(colorama.Fore.WHITE + "----------------------------------------")
        df, groupdict = gasflux.processing.monotonic_transect_groups(df)
        last_transect = None
        last_group_trend = None
        for group in df["group"].unique():
            # leaving this out for now as the group folder is a good identifier
            # if len(df['group'].unique()) == 1:
            #     print(colorama.Fore.RED + f'Skipping{path.name} - only one group')
            #     continue
            group_df = df[df["group"] == group]
            avg_altitudes = group_df.groupby("transect")["altitude"].mean().values
            avg_change = (
                sum([avg_altitudes[i + 1] - avg_altitudes[i] for i in range(len(avg_altitudes) - 1)])
                / len(avg_altitudes)
                if len(avg_altitudes) > 1
                else 0
            )
            # what's the trend
            current_group_trend = "ascending" if avg_change > 0 else "descending"
            # add last transect from previous group if there's a change in trend
            if last_group_trend and last_group_trend != current_group_trend and last_transect is not None:
                group_df = pd.concat([last_transect, group_df])
                avg_altitudes = group_df.groupby("transect")["altitude"].mean().values
            avg_altitudes = np.array(avg_altitudes)
            # check if the group is monotonic
            is_monotonic = np.all(np.diff(avg_altitudes) > 0) or np.all(np.diff(avg_altitudes) < 0)
            if not is_monotonic:  # exception
                print(f"group {group} is not monotonic - check the code!")
            formatted_avg_altitudes = ", ".join([f"{alt:.1f}" for alt in avg_altitudes])
            # do it where transect is the maximum transect number
            last_transect = group_df[group_df["transect"] == group_df["transect"].max()].copy()
            last_transect.loc[:, "transect"] = 0
            last_group_trend = current_group_trend
            unique_transects = len(group_df["transect"].unique())
            if unique_transects < 3:
                print(
                    colorama.Fore.RED
                    + f"Not saving {group} - not enough transects ({unique_transects} transects \
                        at {formatted_avg_altitudes}m)"
                )
            else:
                path = Path(path)
                if output_dir:
                    output_path = Path(
                        Path(output_dir)
                        / f"{path.parents[1].name}"
                        / f"{path.parent.name}_{group}"
                        / f"{path.stem}_{group}.csv"
                    )
                else:
                    # assuming in date and time folder
                    output_path = Path(
                        path.parent.parent.parent.parent
                        / "splits"
                        / path.parent.parent.name
                        / f"{path.parent.name}_{group}"
                        / f"{path.stem}_{group}.csv"
                    )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                group_df.to_csv(output_path, index=False)
                print(
                    colorama.Fore.GREEN
                    + f'wrote {unique_transects} monotonic transects at {formatted_avg_altitudes}m to \n'
                    f'{"/".join(output_path.parts[-5:])}'
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", help="the directory to search for csvs", default="survey")
    parser.add_argument("--search-string", help="the string to search for in the directory", default="*filtered.csv")
    parser.add_argument("--output-dir", help="the metadirectory to save the date/time/csvs to", required=False)
    args = parser.parse_args()
    main(args.target_dir, args.search_string, args.output_dir)
