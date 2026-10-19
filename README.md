*This project has been created as part of the 42 curriculum by akaarich.*

# Call Me Maybe

## Description

**Call Me Maybe** is a Python project that converts natural-language requests into structured JSON function calls using **Qwen3-0.6B** and constrained decoding.

Given a user request and a list of available functions, the program uses the language model to select the most appropriate function and generate its arguments according to the function's schema.

The goal is to produce reliable, schema-compliant JSON by controlling which tokens the language model can generate.

**The program does not execute functions or calculate their results.** It only identifies the function and extracts the required arguments.

### Features

- Function selection using Qwen3-0.6B.
- Constrained decoding with logit masking.
- Automatic fallback selection through `fn_unknown`.
- Support for `number`, `integer`, `string`, and `boolean` parameters.
- Input and output validation.
- Error handling for invalid inputs and unsupported types.
- JSON output generation.

## Instructions

### Requirements

- Python 3.10, 3.11, or 3.12
- `uv` package manager
- Sufficient memory and disk space to load Qwen3-0.6B

### Installation

Clone the repository:

```bash
git clone https://github.com/mool-djaj/Call-me-maybe.git
cd Call-me-maybe
```

Install the dependencies:

```bash
make install
```

This runs `uv sync` to install the project's dependencies.

The model may be downloaded automatically on the first execution and cached for later use.

### Execution

Run the program with the default input files:

```bash
make run
```

Alternatively, execute it directly:

```bash
uv run python -m src
```

You can also specify custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The default paths are:

| File | Path |
|---|---|
| Function definitions | `data/input/functions_definition.json` |
| User requests | `data/input/function_calling_tests.json` |
| Generated results | `data/output/function_calls.json` |

### Development commands

```bash
make lint
```

Runs Flake8 and mypy for static code checks.

```bash
make debug
```

Starts the program using Python's debugger.

```bash
make clean
```

Removes Python cache files and selected development caches.

## Algorithm Explanation

The implementation uses **constrained decoding**, where the language model generates tokens one at a time, but only tokens that maintain a valid function call are permitted.

The process is divided into five main stages.

### 1. Loading and validating inputs

The program reads two JSON files:

- `functions_definition.json`, containing the available function definitions.
- `function_calling_tests.json`, containing the natural-language requests.

Pydantic models validate the input structure and required fields.

The program also loads an internal fallback function, `fn_unknown`, from `data/input/fallback_function.json`.

The Qwen3-0.6B model is initialized using the provided `llm_sdk`.

Function names are tokenized once, and the model's vocabulary is loaded for use during decoding.

### 2. Function selection

The `build_prompt()` function constructs a prompt containing the available functions, their descriptions, parameter types, and the user's request.

The model must choose a function based on the requested operation rather than simply matching words.

The selection process works as follows:

1. `get_allowed_next_ids()` finds the token IDs that can continue an available function name.
2. Qwen generates logits for the next token.
3. `mask_logits()` replaces the logits of invalid tokens with negative infinity (`-inf`).
4. `adjust_fallback_score()` applies a penalty to the fallback continuation when appropriate.
5. The decoder selects the allowed token with the highest score.
6. The process repeats until a complete function name is selected.

For example, if the available functions are:

```text
fn_add_numbers
fn_greet
fn_reverse_string
```

The model cannot generate an unrelated function name because tokens that cannot form one of these names are blocked.

If only one next token is possible, the decoder adds it without requesting new logits.

The algorithm uses **greedy decoding**, selecting the highest-scoring allowed token at each step.

### 3. Fallback handling

The implementation includes an internal function named `fn_unknown`.

Its purpose is to handle requests that cannot be performed by any available function.

The selection prompt instructs the model to choose a real function whenever one matches the requested operation, and to use `fn_unknown` otherwise.

A fallback penalty of `2.5` is applied when the fallback's next token differs from the available real-function continuations.

This reduces the fallback's preference without completely preventing its selection.

If `fn_unknown` is selected, the request is reported as unsuccessful rather than saved as a valid function call.

### 4. Parameter extraction

After selecting a function, `extract_parameters()` generates its arguments one by one.

For each parameter, `build_parameter_prompt()` creates a prompt using:

- The original user request.
- The selected function's signature.
- The current parameter name.
- Previously extracted argument values.

The decoder is chosen according to the parameter type.

| Parameter type | Decoder | Behavior |
|---|---|---|
| `number` | `decode_number()` | Generates a valid JSON number |
| `integer` | `decode_number()` | Generates an integer without decimal or exponent forms |
| `string` | `decode_string()` | Generates valid JSON string content |
| `boolean` | `decode_boolean()` | Chooses between `true` and `false` |

**Numbers:** The decoder uses a validator to check whether the generated text is a valid or incomplete numeric prefix. This allows values such as negative numbers, decimals, and scientific notation while preventing invalid continuations.

**Strings:** The decoder checks candidate tokens to preserve valid JSON string syntax, including escaped characters and closing quotation marks.

**Booleans:** The decoder restricts generation to the token sequences corresponding to `true` and `false`.

Previously extracted parameters are included in subsequent prompts to help maintain consistency when a function requires multiple arguments.

### 5. Output validation and generation

The `validate_result()` function verifies that:

- All required parameters are present.
- No unexpected parameter names are included.
- Every argument has the correct type.
- Numeric values are finite.

The program constructs the final JSON object in Python rather than allowing Qwen to freely generate the entire JSON structure.

Results are serialized using `json.dump()` with `allow_nan=False`.

The saved file is read back for verification and written through a temporary file followed by an atomic replacement.

Each successful function call contains exactly:

```json
{
  "prompt": "User request",
  "name": "function_name",
  "parameters": {}
}
```

## Design Decisions

### Constrained decoding instead of unrestricted generation

Unrestricted text generation cannot guarantee that the model will return a valid function name or correctly formatted JSON.

Constrained decoding limits generation to valid possibilities while still allowing the language model to make the selection.

### Separate function selection and parameter extraction

The program first selects a function, then generates its arguments.

This allows each stage to use instructions and validation rules specific to its task.

### Type-specific decoders

Different data types have different syntax requirements.

Dedicated decoders make it easier to enforce valid numbers, integers, strings, and booleans.

### Pydantic validation

Pydantic is used to validate function definitions and input prompts.

Generated results are checked against the selected function's parameter schema before being saved.

### Reusing model resources

The model is initialized once, function names are tokenized once, and the vocabulary is loaded once.

These resources are reused across user requests to avoid unnecessary repeated setup.

### Using the provided SDK

The implementation interacts with Qwen through the supplied `llm_sdk` methods:

- `encode()`
- `decode()`
- `get_logits_from_input_ids()`
- `get_path_to_vocab_file()`

The project implementation does not directly import PyTorch, Transformers, Outlines, or DSPy.

## Performance Analysis

The project subject specifies the following performance targets:

| Metric | Subject target |
|---|---|
| Function selection and argument accuracy | At least 90% |
| JSON validity | 100% |
| Execution time | Under 5 minutes |
| Error handling | Graceful handling of invalid input and edge cases |

### Accuracy

Function selection is guided by the language model and constrained to the available function names.

Parameter extraction uses type-specific validation to reduce structurally invalid results.

However, constrained decoding does not guarantee that the model chooses the semantically correct function or extracts the correct values.

Accuracy must therefore be measured by comparing generated results against expected function names and arguments on a labeled dataset.

No measured overall accuracy percentage is claimed here.

### JSON reliability

The application constructs the output structure in Python and validates generated arguments before saving them.

This design prevents unrestricted model-generated text from being written directly as the output JSON.

Successfully saved results are serialized and checked for JSON consistency.

Requests that fail validation are reported rather than included as successful function calls.

### Execution speed

The main computational cost comes from repeated calls to `get_logits_from_input_ids()`.

Performance is affected by:

- Model initialization.
- Number of requests.
- Input prompt length.
- Number of generated tokens.
- Available CPU/GPU resources.

The implementation reduces some overhead by reusing the initialized model and vocabulary, and skipping model inference when a function-name continuation has only one possible token.

Numeric generation is limited to 40 tokens, string generation to 64 tokens, and boolean generation to 12 tokens.

The application displays its total processing time after execution.

The subject's five-minute target must be verified using actual execution measurements on the evaluation machine.

### Reliability

The program handles missing or malformed JSON input, invalid function definitions, unsupported parameter types, and decoding failures with descriptive errors.

If an individual request fails, the program continues processing the remaining requests.

Successful calls are saved when available. The process returns a nonzero exit status if one or more requests failed.

## Challenges Faced

### 1. Preventing invalid function names

**Challenge:** A language model can generate arbitrary text rather than choosing from the supplied function definitions.

**Approach:** Function names are converted to token sequences, and prefix matching is used to determine valid next tokens. Invalid logits are masked with negative infinity.

### 2. Handling incomplete numbers

**Challenge:** During generation, values such as `-`, `3.`, or `2e` may be incomplete but still represent valid prefixes of a number.

**Approach:** Numeric validation distinguishes between valid, incomplete, and invalid prefixes so generation can continue until a complete number is produced.

### 3. Generating JSON strings

**Challenge:** Strings may contain quotation marks, backslashes, escaped characters, and tokens that include multiple characters.

**Approach:** The string decoder validates candidate continuations and checks for a correctly completed JSON string before returning its value.

### 4. Extracting multiple parameters

**Challenge:** Generating parameters independently can make it difficult to preserve the context of previously extracted values.

**Approach:** The parameter prompt builder includes previously generated arguments when requesting the next argument.

### 5. Handling unmatched requests

**Challenge:** The model might select a related function even when no available function performs the requested operation.

**Approach:** The implementation includes `fn_unknown` and instructions to select functions based on their actual behavior. A fallback score adjustment helps balance fallback selection against real alternatives.

### 6. Handling invalid inputs and generation failures

**Challenge:** Missing files, malformed JSON, incorrect schemas, or generation limits can interrupt processing.

**Approach:** The application uses validation, exception handling, descriptive error messages, and bounded decoding loops.

## Testing Strategy

The implementation can be validated using the supplied example inputs, custom function definitions, and additional edge cases.

### 1. Input validation

Check how the program handles:

- Missing input files.
- Malformed JSON.
- Invalid function definitions.
- Duplicate function names.
- Empty requests.

The expected behavior is a clear error message rather than an unexpected crash.

### 2. Function selection

Test requests corresponding to the provided functions:

- Addition.
- Greeting.
- String reversal.
- Square root.
- String substitution using a regular expression.

Also test requests that do not match any available function to evaluate fallback selection.

Function names should always come from the supplied definitions.

### 3. Parameter validation

Test parameter extraction using:

- Positive and negative numbers.
- Large numbers and decimal values.
- Empty strings.
- Strings containing spaces, quotes, and special characters.
- Boolean values.
- Functions requiring multiple parameters.

Verify that the returned argument names and types match the selected function definition.

### 4. JSON output verification

After execution, verify that:

1. The output file is valid JSON.
2. Its top-level value is an array.
3. Each result contains exactly `prompt`, `name`, and `parameters`.
4. Parameter names and types match the selected function definition.
5. No extra prose or invalid JSON values are present.

A basic syntax check can be run with:

```bash
python -m json.tool data/output/function_calls.json
```

This checks JSON syntax; schema compliance and semantic correctness require additional checks.

### 5. Accuracy and runtime evaluation

To measure accuracy, use a labeled dataset containing expected function names and argument values.

Compare each generated function call with its expected result and calculate:

```text
Accuracy = Correct function calls / Total requests × 100
```

A correct function call must include both the expected function name and correct argument values.

Measure total runtime using the execution timer displayed by the application.

The provided examples serve as integration checks but are not sufficient evidence of 90% accuracy on unseen requests.

## Example Usage

### Example 1: Addition

User request:

```text
What is the sum of 2 and 3?
```

Expected function-call structure:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

### Example 2: String reversal

User request:

```text
Reverse the string 'hello'
```

Expected function-call structure:

```json
{
  "prompt": "Reverse the string 'hello'",
  "name": "fn_reverse_string",
  "parameters": {
    "s": "hello"
  }
}
```

These examples illustrate expected results, not independently verified outputs from a particular model run.

### Running custom tests

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The output file contains an array of successful function calls.

## Project Structure

```text
Call-me-maybe/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── json_to_token.py
│   ├── function_selector.py
│   ├── prompt_builder.py
│   ├── decoder.py
│   ├── data_type_validator.py
│   ├── function_call.py
│   └── vocab.py
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   ├── function_calling_tests.json
│   │   └── fallback_function.json
│   └── examples/
├── llm_sdk/
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

### Main source files

| File | Responsibility |
|---|---|
| `__main__.py` | Command-line arguments and main execution loop |
| `json_to_token.py` | Input models, JSON validation, and function-name tokenization |
| `function_selector.py` | Function selection and fallback scoring |
| `prompt_builder.py` | Function selection and parameter prompts |
| `decoder.py` | Constrained token decoding |
| `data_type_validator.py` | Numeric and string validation |
| `function_call.py` | Parameter extraction, result validation, and output writing |
| `vocab.py` | Tokenizer vocabulary loading |

## Limitations

- Only `number`, `integer`, `string`, and `boolean` parameter types are supported.
- Nested arrays and objects are not implemented.
- The model may choose an incorrect function or argument value despite syntactic constraints.
- Long string generation may exceed the configured token limit.
- Repeated model inference can be computationally expensive.
- Failed requests are not included in the saved successful results.
- The application generates function calls but does not execute them.

## Resources

### Documentation and references

- Call Me Maybe — 42 project subject.
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python Documentation](https://docs.python.org/3/)
- [Python JSON Documentation](https://docs.python.org/3/library/json.html)
- [Python argparse Documentation](https://docs.python.org/3/library/argparse.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Hugging Face Tokenizer Documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer)
- [uv Documentation](https://docs.astral.sh/uv/)

### AI Usage

AI tools were used as learning and development assistance, including:

- Understanding tokenization, token IDs, logits, and constrained decoding.
- Exploring Python concepts and debugging approaches.
- Suggesting test cases and implementation improvements.
