import json
from pathlib import Path
from typing import cast
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


def load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    """Return a mapping from tokenizer vocabulary text to token ID."""
    path = Path(model.get_path_to_vocab_file())
    try:
        with path.open("r", encoding="utf-8") as file:
            data: object = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load vocabulary: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("Vocabulary must be a dictionary")

    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError("Token must be a string")

        if type(value) is not int:
            raise ValueError("Token ID must be an integer")

    return cast(dict[str, int], data)
