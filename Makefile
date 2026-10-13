.PHONY: install run debug clean lint test

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find src -type d -name '__pycache__' -prune -exec rm -rf {} +
	find src -type f -name '*.pyc' -delete
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 . --exclude=.venv,llm_sdk,__pycache__,.git,./moulinette
	uv run mypy . --exclude='(^|/)(\.venv|llm_sdk|moulinette)/' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs