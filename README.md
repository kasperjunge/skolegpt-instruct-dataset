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
## Dataset Description
### Instruction Pre-and Postfixes

Normalized Frequency	Frequency	Term
6.48517%	226981	'Question:'
0.65466%	22913	'Definition:'
1.30737%	45758	'Detailed Instructions:'
0.65511%	22929	'Instructions:'
7.51428%	263000	'Q:'
0.65137%	22798	'Teacher:'