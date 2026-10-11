"""Run constrained function calling and save schema-checked JSON output."""

import argparse
import json
import math
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

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
from .prompt_builder import (
    build_parameter_prompt,
    build_prompt,
)
from .vocab import load_vocab


ParameterValue = int | float | str | bool
FALLBACK_NAME = "fn_unknown"


def adjust_fallback_score(
    masked: list[float],
    generated: list[int],
    function_tokens: dict[str, list[int]],
    penalty: float,
) -> None:
    """Apply a prior penalty at a branch unique to the fallback."""
    fallback_tokens = function_tokens.get(FALLBACK_NAME)
    if fallback_tokens is None:
        return

    n = len(generated)
    if len(fallback_tokens) <= n or fallback_tokens[:n] != generated:
        return

    fallback_next = fallback_tokens[n]
    real_choices = {
        tokens[n]
        for name, tokens in function_tokens.items()
        if name != FALLBACK_NAME
        and len(tokens) > n
        and tokens[:n] == generated
    }

    # Penalize only when another real candidate is available,
    # and the fallback has its own distinct continuation token.
    if real_choices and fallback_next not in real_choices:
        masked[fallback_next] -= penalty


def select_function(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    prompt: Prompt,
    function_tokens: dict[str, list[int]],
    debug: bool = False,
    fallback_penalty: float = 0.0,
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
        adjust_fallback_score(
            masked, generated, function_tokens, fallback_penalty
        )

        if debug:
            print(f"  Selection prefix: {model.decode(generated)!r}")
            for token_id in allowed_ids:
                candidates = [
                    name
                    for name, ids in function_tokens.items()
                    if len(ids) > len(generated)
                    and ids[:len(generated)] == generated
                    and ids[len(generated)] == token_id
                ]
                if completed and token_id == end_id:
                    candidates = completed
                print(
                    f"    token {token_id}: {logits[token_id]:.3f} "
                    f"(adjusted {masked[token_id]:.3f}) "
                    f"-> {', '.join(candidates)}"
                )

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

        value: ParameterValue

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
            raise ValueError(
                f"Unsupported parameter type: {definition.type!r}"
            )

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
    parser.add_argument(
        "--debug-selection", action="store_true",
        help="Print the model's allowed function-token scores",
    )
    parser.add_argument(
        "--fallback-penalty", type=float, default=1.25,
        help="Experimental logit penalty for fn_unknown (default 1.25)",
    )
    args = parser.parse_args()

    try:
        if not math.isfinite(args.fallback_penalty):
            raise ValueError("Fallback penalty must be finite")
        if args.fallback_penalty < 0:
            raise ValueError("Fallback penalty must not be negative")
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)

        if not functions:
            raise ValueError("Function definitions cannot be empty")
        if len({function.name for function in functions}) != len(functions):
            raise ValueError("Function names must be unique")

        # Keep the fallback separate from the user-provided definitions.
        # It is only a candidate for internal function selection.
        if any(function.name == FALLBACK_NAME for function in functions):
            raise ValueError("Reserved function name: fn_unknown")

        fallback = load_functions("data/input/fallback_function.json")
        if len(fallback) != 1 or fallback[0].name != FALLBACK_NAME:
            raise ValueError("Fallback file must contain only fn_unknown")
        if fallback[0].parameters:
            raise ValueError("fn_unknown must not have parameters")

        selection_functions = functions + fallback
        model = Small_LLM_Model()
        function_tokens = tokenize_function_names(selection_functions, model)
        functions_by_name = {function.name: function for function in functions}
        vocab = {} if args.select_only else load_vocab(model)

        results: list[dict[str, object]] = []
        failures = 0

        for index, prompt in enumerate(prompts, start=1):
            print(f"\n[{index}/{len(prompts)}] {prompt.prompt}")
            try:
                selected_name = select_function(
                    model, selection_functions, prompt, function_tokens,
                    debug=args.debug_selection,
                    fallback_penalty=args.fallback_penalty,
                )
                if selected_name == FALLBACK_NAME:
                    raise ValueError(
                        "No matching function available for this request"
                    )
                selected_function = functions_by_name[selected_name]
                print(f"  Function: {selected_name}")

                if args.select_only:
                    continue

                parameters = extract_parameters(
                    model, selected_function, prompt, vocab
                )
                result = validate_result(
                    prompt, selected_function, parameters
                )
                results.append(result)
                print("  Schema: valid")
            except (
                ValueError,
                TypeError,
                OverflowError,
                RuntimeError,
            ) as error:
                failures += 1
                print(f"  Failed: {error}", file=sys.stderr)

        print(f"\nCompleted: {len(prompts) - failures}/{len(prompts)}")
        if failures:
            if not args.select_only:
                # Prevent an old result from masquerading as this run's output.
                Path(args.output).unlink(missing_ok=True)
            print("No JSON output saved because some prompts failed.")
            return 1

        if args.select_only:
            print("Selection-only test complete; no JSON file written.")
            return 0

        save_results(args.output, results)
        return 0

    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        OverflowError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
