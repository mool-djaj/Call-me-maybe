"""Extract typed arguments, validate results, and save JSON output."""

import json
import math
from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .decoder import decode_boolean, decode_number, decode_string
from .json_to_token import FunctionDefinition, Prompt
from .prompt_builder import build_parameter_prompt

ParameterValue = int | float | str | bool


def extract_parameters(
    model: Small_LLM_Model,
    function: FunctionDefinition,
    prompt: Prompt,
    vocab: dict[str, int],
) -> dict[str, ParameterValue]:
    """Decode arguments according to each parameter's declared type."""
    parameters: dict[str, ParameterValue] = {}
    items = list(function.parameters.items())

    for index, (name, definition) in enumerate(items):
        text = build_parameter_prompt(function, prompt, name, parameters)
        value: ParameterValue

        if definition.type in ("number", "integer"):
            ending = ")" if index == len(items) - 1 else ","
            value = decode_number(
                model, text, vocab, ending,
                integer_only=definition.type == "integer",
            )
        elif definition.type == "string":
            value = decode_string(model, text, vocab)
        elif definition.type == "boolean":
            value = decode_boolean(model, text)
        else:
            raise ValueError(
                f"Unsupported parameter type {definition.type!r} "
                f"for {name!r}"
            )

        parameters[name] = value
        print(f"    {name} = {value!r}")

    return parameters


def validate_result(
    prompt: Prompt,
    function: FunctionDefinition,
    parameters: dict[str, ParameterValue],
) -> dict[str, object]:
    """Confirm argument names and types match the function schema."""
    if set(parameters) != set(function.parameters):
        raise ValueError("Generated parameter names do not match the schema")

    for name, definition in function.parameters.items():
        value = parameters[name]
        if definition.type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Parameter {name!r} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"Parameter {name!r} must be a finite number")
        elif definition.type == "integer":
            if type(value) is not int:
                raise ValueError(f"Parameter {name!r} must be an integer")
        elif definition.type == "string":
            if not isinstance(value, str):
                raise ValueError(f"Parameter {name!r} must be a string")
        elif definition.type == "boolean":
            if type(value) is not bool:
                raise ValueError(f"Parameter {name!r} must be a boolean")
        else:
            raise ValueError(
                f"Unsupported parameter type: {definition.type!r}"
            )

    return {
        "prompt": prompt.prompt,
        "name": function.name,
        "parameters": parameters,
    }


def save_results(path: str, results: list[dict[str, object]]) -> None:
    """Write a complete JSON array atomically and verify its contents."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                results, file, indent=2,
                ensure_ascii=False, allow_nan=False,
            )
            file.write("\n")
        with temporary.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        if saved != results:
            raise ValueError("Saved JSON does not match generated results")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"\nSaved {len(results)} calls to {output}")
