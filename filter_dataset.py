import pathlib

import typer

from skolegpt_instruct_dataset.config import config
from skolegpt_instruct_dataset.filtering import filter_data
from skolegpt_instruct_dataset.utils import load_parquet_file_with_polars


def main(
    n_total: int = config.n_total,
    seed: int = config.seed,
):
    df = load_parquet_file_with_polars(
        config.data_dir / config.sampled_dataset_file_name
    )

    df = filter_data(
        df=df,
        n_total=n_total,
        instruction_sources=config.instruction_sources,
        common_postfixes=config.common_postfixes,
        common_prefixes=config.common_prefixes,
        seed=seed,
    )

    df.write_parquet(config.data_dir / config.filtered_dataset_file_name)


if __name__ == "__main__":
    typer.run(main)
