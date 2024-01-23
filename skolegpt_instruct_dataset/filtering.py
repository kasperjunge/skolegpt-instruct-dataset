import re

import polars as pl

from .utils import return_filter_char_list


def filter_data(
    df: pl.DataFrame,
    n_total: int,
    instruction_sources: list[str],
    common_prefixes: list[str],
    common_postfixes: list[str],
    seed: int,
) -> pl.DataFrame:
    """Preprocess raw OpenOrca dataset."""

    original_dataset_size = len(df)
    print(
        "Starting filter_data function. Original dataset size:", original_dataset_size
    )

    df = remove_translation_instructions(df)
    print("Completed removing translation instructions.")

    df = remove_common_pre_postfixes(df, common_prefixes, common_postfixes)
    print("Completed removing common prefixes and postfixes.")

    df = remove_questions_ending_with_colon(df)
    print("Completed removing questions ending with a colon.")

    df = remove_multiple_choice_questions(df)
    print("Completed removing multiple choice questions.")

    df = basic_cleaning(df)
    print("Completed basic cleaning.")

    df = remove_exotic_chars(df)
    print("Completed removing examples with exotic characters.")

    report_percent_removed(df, original_dataset_size)

    return df


# ---------------------------------------------------------------------------- #
#                              Preprocessing Steps                             #
# ---------------------------------------------------------------------------- #


def remove_translation_instructions(df):
    # Hard filter on "translate"
    df = df.filter(~df["question"].str.to_lowercase().str.contains("translate"))
    return df


def remove_common_pre_postfixes(df, common_prefixes, common_postfixes):
    prefix_pattern = r"^(?:" + "|".join(re.escape(p) for p in common_prefixes) + ")"
    postfix_pattern = r"(?:" + "|".join(re.escape(p) for p in common_postfixes) + ")$"
    df = df.with_columns(
        df["question"].str.replace_all(prefix_pattern, "").alias("question")
    )
    return df.with_columns(
        df["question"].str.replace_all(postfix_pattern, "").alias("question")
    )


def remove_questions_ending_with_colon(df):
    return df.filter(~df["question"].map_elements(lambda x: x.strip().endswith(":")))


def remove_multiple_choice_questions(df):
    option_patterns = [
        r"(?i)\b[A-D]\)",  # Matches A), B), C), D) in a case-insensitive manner
        r"(?i)\b[1-4]\)",  # Matches 1), 2), 3), 4) in a case-insensitive manner
        r"(?i)\b\([A-D]\)",  # Matches (A), (B), (C), (D) in a case-insensitive manner
        r"(?i)\b[A-D]\.",  # Matches A., B., C., D. in a case-insensitive manner
        r"(?i)\b[A-D]:",  # Matches A:, B:, C:, D: in a case-insensitive manner
        r"\(i+\)",  # Matches (i), (ii), (iii), etc.
        r"\[[A-Z]\]",  # Matches [A], [B], [C], etc.
        r"\b[i]+\.",  # Matches ii., iii., iv., etc.
    ]
    combined_option_pattern = "|".join(option_patterns)
    condition = (
        df["question"].str.contains("Options:")
        | df["question"].str.contains("OPT:")
        | df["question"].str.contains("OPTIONS:")
    ) | df["question"].str.contains(combined_option_pattern)
    return df.filter(~condition)


def basic_cleaning(df):
    df = df.with_columns(
        [
            df["system_prompt"].str.strip_chars(),
            df["question"].str.strip_chars(),
            df["response"].str.strip_chars(),
        ]
    )
    df = df.filter(df["question"] != "")
    df = df.filter(df["response"] != "")
    return df


def report_percent_removed(df, original_size):
    percent_removed = round(100 * (1 - len(df) / original_size), 4)
    print(f"{percent_removed} % of dataset removed after preprocessing.")


def stratify_dataframe(
    df: pl.DataFrame,
    n_total: int,
    instruction_sources: list[str],
    seed: int,
) -> pl.DataFrame:
    """Stratify instruction examples according to instruction source."""
    n_sources = len(instruction_sources)
    samples_per_source = n_total // n_sources

    sampled_dfs = []
    remaining_n_total = n_total

    # First, sample from sources with enough data
    for source in instruction_sources:
        source_df = df.filter(df["source"] == source)
        n_source = source_df.height
        n_sample = min(samples_per_source, n_source)
        sampled_dfs.append(source_df.sample(n_sample, seed=seed))
        remaining_n_total -= n_sample

    # Distribute remaining samples if necessary
    if remaining_n_total > 0:
        remaining_sources = [
            source
            for source in instruction_sources
            if df.filter(df["source"] == source).height > samples_per_source
        ]
        additional_samples_per_source = (
            remaining_n_total // len(remaining_sources) if remaining_sources else 0
        )

        for source in remaining_sources:
            source_df = df.filter(df["source"] == source)
            n_source = source_df.height - samples_per_source
            n_additional_sample = min(additional_samples_per_source, n_source)
            sampled_dfs.append(source_df.sample(n_additional_sample, seed=seed))

    df = pl.concat(sampled_dfs)

    return df


def remove_exotic_chars(df: pl.DataFrame) -> pl.DataFrame:
    # Filter questions and responses with exotic chars
    def contains_characters(text, characters):
        for char in characters:
            if char in text:
                return True
        return False

    filter_chars = return_filter_char_list(df=df)

    df = df.filter(
        ~df["question"].map_elements(
            lambda x: contains_characters(text=x, characters=filter_chars)
        )
    )
    df = df.filter(
        ~df["response"].map_elements(
            lambda x: contains_characters(text=x, characters=filter_chars)
        )
    )
    return df
