.PHONY: install run debug clean lint mac moulinette

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

mac:
	mkdir -p /goinfre/$USER/.cache/huggingface
	export HF_HOME="/goinfre/$USER/.cache/huggingface"
	export HF_HUB_CACHE="$HF_HOME/hub"
	export HF_XET_CACHE="$HF_HOME/xet"

SET = public
MOU_DIR = data_moulinette/$(SET)

moulinette:
	cd moulinette && uv run python -m moulinette prepare_exercises \
		--output ../$(MOU_DIR) --set $(SET)
	rm -f $(MOU_DIR)/output/function_calls.json
	uv run python -m src \
		--functions_definition $(MOU_DIR)/input/functions_definition.json \
		--input $(MOU_DIR)/input/function_calling_tests.json \
		--output $(MOU_DIR)/output/function_calls.json \
		|| echo "Some prompts failed; grading saved results."
	test -f $(MOU_DIR)/output/function_calls.json
	cd moulinette && uv run python -m moulinette grade_student_answers \
		../$(MOU_DIR)/output/function_calls.json --set $(SET)
