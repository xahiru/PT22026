# Stock Price Forecasting - project commands
# On Windows without Make, run the equivalent `python` / `pytest` commands
# listed in the README directly.

.PHONY: setup run train figures test clean

setup:
	python -m pip install -r requirements.txt

run:
	python app.py

train:
	python scripts/run_pipeline.py

figures:
	python scripts/make_figures.py

test:
	pytest

clean:
	rm -rf results/figures __pycache__ .pytest_cache tests/__pycache__
