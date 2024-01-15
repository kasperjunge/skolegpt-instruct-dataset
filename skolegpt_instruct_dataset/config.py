class Config:
    seed: int = 42
    n_max: int = 3000000  # number of raw samples from OpenOrca dataset
    n_total: int = 90000  # number of examples in final Danish translated dataset
    instruction_sources: list[str] = [
        "flan",
        "niv",
        "t0",
        "cot",
    ]  # instuction example sources
    common_prefixes: list[str] = [
        "Question:",
        "Definition:",
        "Detailed Instructions:",
        "Instructions:",
        "Q:",
        "Teacher:",
    ]
    common_postfixes: list[str] = [
        "Answer:",
        "Solution:",
        "A:",
        "Output:",
        "Student:",
    ]


config = Config()
