import polars as pl


def preprocess_data(
    df: pl.DataFrame,
    n_total: int,
    instruction_sources: list[str],
    common_prefixes: list[str],
    common_postfixes: list[str],
    seed: int,
) -> pl.DataFrame:
    """Preprocess data sample from the OpenOrca dataset."""

    # ---------------------------------------------------------------------------- #
    #                               Filter Questions                               #
    # ---------------------------------------------------------------------------- #

    # Attempt to remove translation instructions
    print("Removing translate instructions..")
    df = df.filter(~df["question"].str.to_lowercase().str.contains("translate"))

    # Remove prefixes from strings
    print("Removing common question prefixes..")
    for prefix in common_prefixes:
        df = df.with_columns(
            df["question"].map_elements(
                lambda x: x[len(prefix) :].strip() if x.startswith(prefix) else x
            )
        )

    # Remove postfixes from strings
    print("Removing common question postfixes..")
    for postfix in common_postfixes:
        df = df.with_columns(
            df["question"].map_elements(
                lambda x: x[: -len(postfix)].strip() if x.endswith(postfix) else x
            )
        )

    # Remove remaining examples with common pre- and postfixes
    print("Removing remaining questions with common pre- and postfixes..")
    prefix_pattern = "^(%s)" % "|".join([prefix.lower() for prefix in common_prefixes])
    postfix_pattern = "(%s)$" % "|".join(
        [postfix.lower() for postfix in common_postfixes]
    )

    for pattern in [prefix_pattern, postfix_pattern]:
        df = df.filter(
            ~df["question"].str.strip_chars().str.to_lowercase().str.contains(pattern)
        )

    # ---------------------------------------------------------------------------- #
    #                        Stratify by Instruction Sources                       #
    # ---------------------------------------------------------------------------- #

    # Stratify by source
    print("Stratifying dataset by instruction source..")
    df = stratify_dataframe(
        df=df,
        n_total=n_total,
        instruction_sources=instruction_sources,
        seed=seed,
    )

    return df


def stratify_dataframe(
    df: pl.DataFrame,
    n_total: int,
    instruction_sources: list[str],
    seed: int,
) -> pl.DataFrame:
    """Stratify instruction examples according to instruction source (niv, flan, t0, cot)."""
    n_sources = len(instruction_sources)

    df = pl.concat(
        [
            df.filter(df["source"] == source).sample(n_total / n_sources, seed=seed)
            for source in instruction_sources
        ]
    )

    return df
