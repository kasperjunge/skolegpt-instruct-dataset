import textwrap

import polars as pl

from .config import config


def sample_and_print_example(df):
    """Sample random example and print it."""

    s = df.sample(1)
    print("ID")
    print(s.get_column("id")[0])
    print()
    print("System Prompt:\n".upper())
    print(textwrap.fill(s.get_column("system_prompt")[0]))
    print()
    print("Question:\n".upper())
    print(textwrap.fill(s.get_column("question")[0]))
    print()
    print("Response:\n".upper())
    print(textwrap.fill(s.get_column("response")[0]))


def analyse_pre_and_postfixes(df: pl.DataFrame):
    """Analyse frequency of common pre- and postfixes."""

    # Analysing common question/instruction pre- and postfixes
    print("--- Prefixes ---")
    for prefix in config.common_prefixes:
        freq = (
            df["question"]
            .str.strip_chars()
            .str.to_lowercase()
            .str.contains("^{prefix}".format(prefix=prefix.lower()))
            .sum()
        )
        norm_freq = round(freq / len(df) * 100, 5)
        print(f"Normalized Freq. {norm_freq}% | Freq.: {freq} | Term: '{prefix}' ")

    print("\n--- Postfixes ---")
    for postfix in config.common_postfixes:
        freq = (
            df["question"]
            .str.strip_chars()
            .str.to_lowercase()
            .str.contains("{prefix}$".format(prefix=postfix.lower()))
            .sum()
        )
        norm_freq = round(freq / len(df) * 100, 5)
        print(f"Normalized Freq. {norm_freq}% | Freq.: {freq} | Term: '{postfix}' ")
