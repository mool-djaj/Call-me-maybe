*This project has been created as part of the 42 curriculum by akaarich.*

# Call Me Maybe

## Description

Call Me Maybe converts a natural-language request into a typed function call using
Qwen/Qwen3-0.6B through the supplied `llm_sdk`. **It does not execute functions**.
The output is an array of JSON objects containing exactly `prompt`, `name`, and
`parameters`, with no additional keys.

## Instructions

Python 3.10+ and `uv` are required. From the repository root:

```bash
make install
make run
```

The first run may download Qwen weights (~1.5 GB). Subsequent runs normally use
Hugging Face's cache. `make run` writes to `data/output/function_calls.json`.

The mandatory command and custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

For diagnostic function selection only:

```bash
uv run python -m src --select-only
```

Tests and style checks:

```bash
make test
make lint
make debug
make clean
```

The `data/output/` folder is intentionally ignored by Git and is created on successful
execution. If any request fails, the program exits nonzero and does not replace the
existing output file.

## Algorithm explanation

1. Pydantic validates the supplied function definitions and input prompts.
2. The model loads once; names are encoded once and kept in memory.
3. For each request, the function-choice prompt is encoded. A prefix-matching
   constrained decoder permits only token IDs continuing a supplied function name.
   All other logits are masked to negative infinity before choosing the highest
   remaining logit. When only one continuation is possible, model inference is
   skipped for that token. If one name is a prefix of another, a newline
   ending is also offered so the LLM can choose the shorter name.
4. After choosing a function, parameter types determine which decoder is used:
   - `number`: JSON-number state machine permits valid prefixes, including unfinished
     minus signs, decimal points, and scientific notation. A delimiter ends the number
     only from a complete state.
   - `integer`: number-like constrained decoding disallowing fractions/exponents.
   - `string`: a state machine permits valid JSON string contents and escapes;
     an unescaped closing double quote ends the string. Candidate token sequences
     are checked after decoding; the highest scoring valid token is selected.
   - `boolean`: constrained selection between the tokenizations of `true` and `false`.
5. Python constructs JSON keys and punctuation deterministically. It validates
   the generated values against the selected function's schema, serializes with
   `json.dump(..., allow_nan=False)`, reads the file back for verification, and
   atomically replaces the requested output file.

The LLM chooses the function and argument values. Python only constrains the
allowed form and constructs the final JSON structure.

## Design decisions

- Uses the supplied public `llm_sdk` methods only: `encode`, `decode`,
  `get_logits_from_input_ids`, and `get_path_to_vocab_file`.
- Does not import PyTorch, Transformers, Outlines or DSPy in project implementation.
  The **provided SDK itself** imports dependencies necessary for model execution.
- No regex is needed for partial number or JSON-string recognition.
- Function names and vocabulary are cached in memory for all input prompts.
- Limits generation length so malformed outputs cannot loop indefinitely.
- Unsupported schema types raise a descriptive error instead of silently fabricating
  invalid JSON; nested arrays/objects are not implemented (optional bonus work).

## Testing strategy

Run `make test` for fast model-free unit tests of validators, masking, decoding,
JSON schema checks, and file output. Run `make run` for actual Qwen integration.
Include cases with multi-digit and negative numbers, decimals/exponents, empty
strings, escaped quotation marks, boolean values, and different function sets.

Note: tests with the supplied five demo prompts are a smoke test, **not proof of
90%+ accuracy on unseen requests**. Test a larger labeled input set to calculate
selection/argument accuracy and time. Performance depends strongly on CPU/GPU
and the number/length of requests, so no unmeasured benchmark is claimed.

## Performance analysis

The main latency comes from `get_logits_from_input_ids`, not from `encode`.
Function-name tokens and the vocabulary are computed/loaded once. The decoder
skips inference on unambiguous function-name continuations. The supplied SDK
re-evaluates the prompt on each next-token call, so processing lengthy strings
may be expensive. Measure runtime against the subject's five-minute requirement
on the evaluation machine.

## Challenges faced

- Unconstrained generation produced non-JSON text and even attempted to answer
  the request rather than emit a function call. Masking fixed structural choices.
- Partial numbers such as `-` and `3.` must not be rejected before completion;
  the state machine keeps valid prefixes rather than checking only finished numbers.
- Extracting the second argument required including previously generated arguments
  in the prompt.
- String parameters need special handling of closing quotes and escape sequences.

## Example usage

Input request: `What is the sum of 2 and 3?`

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2, "b": 3}
}
```

## Resources and AI use

- Project subject, *Call Me Maybe*, v1.7.
- Python `json` documentation: https://docs.python.org/3/library/json.html
- Python `argparse` documentation: https://docs.python.org/3/library/argparse.html
- Pydantic documentation: https://docs.pydantic.dev/
- Hugging Face tokenizer documentation: https://huggingface.co/docs/transformers/main_classes/tokenizer
- AI was used to explain tokenization, logarithmic scores/logit masking,
  build prototypes and test fixtures, and help draft documentation. The final
  implementation should be understood, manually reviewed, and checked by peers.

### Optional additional type smoke test

A separate demonstration input set is in `data/examples/`. Its results are not
claimed as verified Qwen outputs. Run it with:

```bash
uv run python -m src \
  --functions_definition data/examples/functions_definition.json \
  --input data/examples/function_calling_tests.json \
  --output data/output/examples.json
```
