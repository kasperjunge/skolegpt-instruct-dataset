import pathlib

import polars as pl

from skolegpt_instruct_dataset.config import config
from skolegpt_instruct_dataset.data import get_data
from skolegpt_instruct_dataset.preprocess import preprocess_data


def main() -> None:
    # params
    cache_file = pathlib.Path(__file__).parent / "orca_sample.parquet"
    use_cache = True

    # get date
    if cache_file.is_file() & use_cache:
        df = pl.read_parquet(cache_file)
    else:
        df = get_data(n_max=config.n_max)
        df.write_parquet(cache_file)

    # preprocess
    df = preprocess_data(
        df=df,
        n_total=config.n_total,
        instruction_sources=config.instruction_sources,
        common_postfixes=config.common_postfixes,
        common_prefixes=config.common_prefixes,
        seed=config.seed,
    )


if __name__ == "__main__":
    main()
