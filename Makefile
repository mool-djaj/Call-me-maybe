.PHONY: install run debug clean lint

install:
	uv sync

help:
	uv run python3 -m src --help
run:
	uv run python3 -m src

debug:
	uv run python3 -m pdb src/__main__.py

clean:
	rm -rf __pycache__ .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f data/output/*.json

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs