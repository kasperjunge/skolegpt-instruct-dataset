# skolegpt-instruct-dataset 🔥

Code for creating an open source Danish instruction dataset for fine-tuing Danish LLM's by translating a subset of the [OpenOrca instruction dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca). The project is a part of the [SkoleGPT project](https://skolegpt.dk/).

The project consists of 3 components:

1. OpenOrca sampling: sample_dataset.py
2. Quality filtering: filter_dataset.py
3. Danish translation: translate_dataset.py (the expensive one 💵)

## Usage
1. Sample OpenOrca dataset:
```bash
poetry run python sample_dataset.py 
```

2. Filter sampled dataset:
```bash
poetry run python filter_dataset.py
```

3. Translate filterdd dataset:
```bash
poetry run python translate_dataset.py
```
# Dataset Description
## Data
The data extraction process involves loading and shuffling the [OpenOrca dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca), specifically the "1M-GPT4-Augmented.parquet" file. A specified number of entries are then selected to form a subset, which is organized into a DataFrame with an added "source" column for origin tracking. This results in a manageable and tailored subset of the dataset for analysis or further processing.

## Filtering
The filter_data function is designed to preprocess and filter the raw OpenOrca dataset. This process involves several steps, each targeting specific types of data or formatting issues within the dataset. 

Below is an outline of these steps:

1. **Remove Translation Instructions:** Filters out entries containing the word "translate" in the "question" field, targeting instances that are likely to be translation instructions.

2. **Remove Common Prefixes and Postfixes:** Strips common prefixes and postfixes from the "question" field. This is achieved through regular expressions constructed from provided lists of common prefixes and postfixes.

3. **Remove Questions Ending with a Colon:** Filters out entries where the "question" field ends with a colon, as these often indicate incomplete or improperly formatted questions.

4. **Remove Multiple Choice Questions:** Identifies and removes multiple-choice questions. This is done using regular expressions to detect common multiple-choice question formats, such as options labeled with letters or numbers.

5. **Basic Cleaning:** Performs basic cleaning of the dataset by stripping characters from the "system_prompt", "question", and "response" fields and removing entries where "question" or "response" fields are empty.

6. **Remove Exotic Characters:** Filters out entries containing exotic characters in the "question" and "response" fields. The list of characters to filter is dynamically generated based on the dataset content.

7. **Remove Duplicate Questions and Responses:** Eliminates duplicates in the dataset, ensuring uniqueness in both "question" and "response" fields.