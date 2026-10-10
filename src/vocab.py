"""Load the vocabulary file exposed by the public LLM SDK."""

import json
from pathlib import Path
from typing import cast

from llm_sdk import Small_LLM_Model


def load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    """Return a mapping from tokenizer vocabulary text to token ID."""
    path = Path(model.get_path_to_vocab_file())
    try:
        with path.open("r", encoding="utf-8") as file:
            data: object = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load vocabulary: {error}") from error
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and type(value) is int
        for key, value in data.items()
    ):
        raise ValueError("Vocabulary is not a token-to-ID dictionary")
    return cast(dict[str, int], data)
