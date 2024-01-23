.PHONY: dataset

dataset:
	@echo "Running sample_dataset.py..."
	poetry run python sample_dataset.py

	@echo "Running filter_dataset.py..."
	poetry run python filter_dataset.py

	@echo "Running stratify_dataset.py..."
	poetry run python stratify_dataset.py

	@echo "Done."
