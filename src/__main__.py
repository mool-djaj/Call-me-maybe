"""CLI entry point for constrained function calling."""

import argparse
import math
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .function_call import extract_parameters, save_results, validate_result
from .function_selector import FALLBACK_NAME, load_candidates, select_function
from .json_to_token import load_prompts, tokenize_function_names
from .vocab import load_vocab


def parse_args() -> argparse.Namespace:
    """Read input/output paths and selection options."""
    parser = argparse.ArgumentParser(description="Constrained function caller")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json"
    )
    parser.add_argument("--output", default="data/output/function_calls.json")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--debug-selection", action="store_true")
    parser.add_argument("--fallback-penalty", type=float, default=1.25)
    return parser.parse_args()


def main() -> int:
    """Select functions, extract arguments, and save valid calls."""
    args = parse_args()
    try:
        if not math.isfinite(args.fallback_penalty):
            raise ValueError("Fallback penalty must be finite")
        if args.fallback_penalty < 0:
            raise ValueError("Fallback penalty must not be negative")

        functions, by_name = load_candidates(args.functions_definition)
        prompts = load_prompts(args.input)
        model = Small_LLM_Model()
        function_tokens = tokenize_function_names(functions, model)
        vocab = {} if args.select_only else load_vocab(model)
        results: list[dict[str, object]] = []
        failures = 0

        for index, prompt in enumerate(prompts, start=1):
            print(f"\n[{index}/{len(prompts)}] {prompt.prompt}")
            try:
                if not prompt.prompt.strip():
                    raise ValueError("Empty user request")

                name = select_function(
                    model, functions, prompt, function_tokens,
                    debug=args.debug_selection,
                    fallback_penalty=args.fallback_penalty,
                )
                if name == FALLBACK_NAME:
                    raise ValueError(
                        "No matching function available for this request"
                    )

                function = by_name[name]
                print(f"  Function: {name}")
                if not args.select_only:
                    parameters = extract_parameters(
                        model, function, prompt, vocab
                    )
                    results.append(
                        validate_result(prompt, function, parameters)
                    )
                    print("  Schema: valid")

            except (
                ValueError, TypeError, OverflowError, RuntimeError
            ) as error:
                failures += 1
                print(f"  Failed: {error}", file=sys.stderr)

        print(f"\nCompleted: {len(prompts) - failures}/{len(prompts)}")
        if failures:
            if not args.select_only:
                Path(args.output).unlink(missing_ok=True)
            print("No JSON output saved because some prompts failed.")
            return 1
        if args.select_only:
            print("Selection-only test complete; no JSON file written.")
            return 0

        save_results(args.output, results)
        return 0

    except (
        OSError, ValueError, TypeError, RuntimeError, OverflowError
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
