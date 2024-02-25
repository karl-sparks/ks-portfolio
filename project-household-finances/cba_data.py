import polars as pl

from typing import List

import os

import config as cfg


def process_cba_transactions(
    input_filepath: str, output_filepath: str, col_names: List[str]
) -> bool:
    if not os.path.exists(input_filepath):
        print(f"Can not find input file {input_filepath}")
        return False

    pl.read_csv(
        source=input_filepath,
        has_header=False,
        new_columns=col_names,
        try_parse_dates=True,
    ).write_parquet(file=output_filepath)

    return os.path.exists(output_filepath)


if __name__ == "__main__":
    process_cba_transactions(cfg.CBA_CSV_FILEPATH, cfg.CBA_FILEPATH, cfg.CBA_COLUMNS)
