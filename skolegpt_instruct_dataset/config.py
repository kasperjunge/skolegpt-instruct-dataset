class Config:
    seed: int = 42
    n_max: int = 3500000  # number of raw samples from OpenOrca dataset
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
        # new
        "Student:",
        "Write a sentence not in English.",
        "Denny asked:",
        "Choose your answer:",
    ]
    common_postfixes: list[str] = [
        "Answer:",
        "Solution:",
        "A:",
        "Output:",
        "Teacher:",
        "Student:",
        # new
        "Stream of thoughts:",
        "Step-by-step reasoning:",
        "Chain-of-thought:",
        "Let's think first:",
        "The thinking starts now:",
        "Stream of consciousness:",
        "Which language is this?",
        "Please think gradually:",
        "The answer is:",
        "Me:",
        "Some thinking first:",
        "Some random thoughts:",
        "Let's solve step-by-step:",
        "Numbered answers:",
        "Let's answer step by step:",
        "The answer to this question is:",
        "Explanation:",
        "Teacher: Let's think:",
        "Let's think:",
        "Chain of thought:",
        "Your thoughts:",
        "Summary:",
    ]


config = Config()
