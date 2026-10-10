"""Load and validate function definitions and user prompts."""

import json
from pathlib import Path

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict, ValidationError


class TypeDefinition(BaseModel):
    """Describe one schema type."""

    model_config = ConfigDict(extra="forbid")
    type: str


class FunctionDefinition(BaseModel):
    """Definition of a callable function and its parameters."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, TypeDefinition]
    returns: TypeDefinition


class Prompt(BaseModel):
    """Single natural-language request."""

    model_config = ConfigDict(extra="forbid")
    prompt: str


def load_json(path: str) -> object:
    """Read a JSON file with a clear error for missing or invalid input."""
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            result: object = json.load(file)
        return result
    except FileNotFoundError as error:
        raise ValueError(f"File not found: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON file {path}: {error}") from error


def load_functions(path: str) -> list[FunctionDefinition]:
    """Validate the function definitions JSON array."""
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Functions file must contain a JSON array")
    try:
        functions = [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as error:
        raise ValueError(f"Invalid function definition: {error}") from error
    if not functions:
        raise ValueError("Function list cannot be empty")
    if any(not item.name for item in functions):
        raise ValueError("Function names must not be empty")
    if len({item.name for item in functions}) != len(functions):
        raise ValueError("Function names must be unique")
    return functions


def load_prompts(path: str) -> list[Prompt]:
    """Validate the prompts JSON array."""
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Prompts file must contain a JSON array")
    try:
        return [Prompt.model_validate(item) for item in data]
    except ValidationError as error:
        raise ValueError(f"Invalid prompt: {error}") from error


def tokenize_function_names(
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
) -> dict[str, list[int]]:
    """Encode function names once and reuse the token sequences."""
    tokens: dict[str, list[int]] = {}
    for function in functions:
        ids: list[int] = model.encode(function.name)[0].tolist()
        if not ids or model.decode(ids) != function.name:
            raise ValueError(f"Could not encode function name {function.name!r}")
        tokens[function.name] = ids
    return tokens
