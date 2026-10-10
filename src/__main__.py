"""Run constrained function calling and save schema-checked JSON output."""

import argparse
import json
import math
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model

from .decoder import (
    decode_number,
    decode_string,
    decode_boolean,
    get_allowed_next_ids,
    mask_logits,
)
from .json_to_token import (
    FunctionDefinition,
    Prompt,
    load_functions,
    load_prompts,
    tokenize_function_names,
)
from .prompt_builder import build_parameter_prompt, build_prompt
from .vocab import load_vocab


ParameterValue = int | float | str | bool


def select_function(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    prompt: Prompt,
    function_tokens: dict[str, list[int]],
) -> str:
    """Select a function using constrained next-token decoding."""
    text = build_prompt(functions, prompt)
    input_ids = model.encode(text)[0].tolist()
    generated: list[int] = []

    end_ids: list[int] = model.encode("\n")[0].tolist()
    if len(end_ids) != 1:
        raise ValueError("Cannot encode function-name ending")
    end_id = end_ids[0]

    while True:
        completed = [
            name for name, ids in function_tokens.items() if ids == generated
        ]
        allowed_ids = get_allowed_next_ids(function_tokens, generated)

        if completed and not allowed_ids:
            return completed[0]
        if not allowed_ids:
            raise ValueError("No matching function continuation")

        # If one name is a token-prefix of another, let Qwen decide
        # between ending the shorter name and extending it.
        if completed:
            allowed_ids.append(end_id)

        if len(allowed_ids) == 1:
            generated.append(allowed_ids[0])
            continue

        logits = model.get_logits_from_input_ids(input_ids + generated)
        masked = mask_logits(logits, allowed_ids)
        next_id = max(allowed_ids, key=lambda token_id: masked[token_id])
        if completed and next_id == end_id:
            return completed[0]
        generated.append(next_id)


def extract_parameters(
    model: Small_LLM_Model,
    function: FunctionDefinition,
    prompt: Prompt,
    vocab: dict[str, int],
) -> dict[str, ParameterValue]:
    """Extract number and string arguments using the appropriate decoder."""
    parameters: dict[str, ParameterValue] = {}
    items = list(function.parameters.items())

    for index, (name, definition) in enumerate(items):
        text = build_parameter_prompt(function, prompt, name, parameters)

        if definition.type in ("number", "integer"):
            is_last = index == len(items) - 1
            end_char = ")" if is_last else ","
            value = decode_number(
                model, text, vocab, end_char,
                integer_only=definition.type == "integer",
            )
        elif definition.type == "string":
            value = decode_string(model, text, vocab)
        elif definition.type == "boolean":
            value = decode_boolean(model, text)
        else:
            raise ValueError(
                f"Unsupported parameter type {definition.type!r} for {name!r}"
            )

        parameters[name] = value
        print(f"    {name} = {value!r}")

    return parameters


def validate_result(
    prompt: Prompt,
    function: FunctionDefinition,
    parameters: dict[str, ParameterValue],
) -> dict[str, object]:
    """Check that generated arguments match the selected function schema."""
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
            raise ValueError(f"Unsupported parameter type: {definition.type!r}")

    return {
        "prompt": prompt.prompt,
        "name": function.name,
        "parameters": parameters,
    }


def save_results(path: str, results: list[dict[str, object]]) -> None:
    """Save a JSON array and verify it can be loaded back successfully."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False,
                      allow_nan=False)
            file.write("\n")

        with temporary_path.open("r", encoding="utf-8") as file:
            saved = json.load(file)

        if saved != results:
            raise ValueError("Saved JSON does not match generated results")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"\nSaved {len(results)} calls to {output_path}")


def main() -> int:
    """Process each input prompt and write the complete output JSON array."""
    parser = argparse.ArgumentParser(description="Constrained function caller")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output", default="data/output/function_calls.json"
    )
    parser.add_argument(
        "--select-only", action="store_true",
        help="Only test function selection; do not create output JSON",
    )
    args = parser.parse_args()

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)

        if not functions:
            raise ValueError("Function definitions cannot be empty")
        if len({function.name for function in functions}) != len(functions):
            raise ValueError("Function names must be unique")

        model = Small_LLM_Model()
        function_tokens = tokenize_function_names(functions, model)
        functions_by_name = {function.name: function for function in functions}
        vocab = {} if args.select_only else load_vocab(model)

        results: list[dict[str, object]] = []
        failures = 0

        for index, prompt in enumerate(prompts, start=1):
            print(f"\n[{index}/{len(prompts)}] {prompt.prompt}")
            try:
                selected_name = select_function(
                    model, functions, prompt, function_tokens
                )
                print(f"  Function: {selected_name}")

                if args.select_only:
                    continue

                selected_function = functions_by_name[selected_name]
                parameters = extract_parameters(
                    model, selected_function, prompt, vocab
                )
                result = validate_result(
                    prompt, selected_function, parameters
                )
                results.append(result)
                print("  Schema: valid")
            except (ValueError, TypeError, OverflowError, RuntimeError) as error:
                failures += 1
                print(f"  Failed: {error}", file=sys.stderr)

        print(f"\nCompleted: {len(prompts) - failures}/{len(prompts)}")
        if failures:
            print("No new output was written; previous output may still exist.")
            return 1

        if args.select_only:
            print("Selection-only test complete; no JSON file written.")
            return 0

        save_results(args.output, results)
        return 0

    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
