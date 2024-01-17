import pathlib
import time

import polars as pl

from skolegpt_instruct_dataset.config import config
from skolegpt_instruct_dataset.data import get_data
from skolegpt_instruct_dataset.preprocess import preprocess_data
from skolegpt_instruct_dataset.utils import print_elapsed_time


def main() -> None:
    # params
    cache_file = pathlib.Path(__file__).parent / "orca_sample.parquet"
    cache_file_preprocessed = (
        pathlib.Path(__file__).parent / "orca_sample_preprocessed.parquet"
    )
    use_cache = True

    # ---------------------------------------------------------------------------- #
    #                                   Get Data                                   #
    # ---------------------------------------------------------------------------- #

    print("Step 1: Get Data..")
    start_time = time.time()
    if cache_file.is_file() & use_cache:
        df = pl.read_parquet(cache_file)
    else:
        df = get_data(n_max=config.n_max)
        df.write_parquet(cache_file)
    print_elapsed_time(step="Get data", start_time=start_time)

    # ---------------------------------------------------------------------------- #
    #                                Preprocess Data                               #
    # ---------------------------------------------------------------------------- #

    print("\nStep 2: Preprocess Data..")
    start_time = time.time()
    if cache_file_preprocessed.is_file() & use_cache:
        df = pl.read_parquet(cache_file_preprocessed)
    else:
        df = preprocess_data(
            df=df,
            n_total=config.n_total,
            instruction_sources=config.instruction_sources,
            common_postfixes=config.common_postfixes,
            common_prefixes=config.common_prefixes,
            seed=config.seed,
        )
        df.write_parquet(cache_file_preprocessed)

    print_elapsed_time(step="Preprocess data", start_time=start_time)


if __name__ == "__main__":
    main()
