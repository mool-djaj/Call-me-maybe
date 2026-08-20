This project has been created as part of the 42 curriculum by akaarich.

## Description
**Call-me-bb** is a function calling tool that translates natural language prompts into strictly structured JSON function calls[cite: 1]. Large Language Models do not naturally produce reliable machine-executable output; this project bridges that gap[cite: 1]. By utilizing a technique called constrained decoding, the engine intercepts the token generation of the Qwen3-0.6B model and guarantees 100% schema-compliant JSON output without relying on generic prompting heuristics[cite: 1].

## Instructions
The project relies on `uv` for strict dependency management[cite: 1].

1. **Install dependencies:**
   ```bash
   make install
   ```
   *(This will run `uv sync` to lock and build the environment, including Pydantic, Numpy, and the necessary PyTorch/Transformers libraries).*

2. **Run the Engine:**
   ```bash
   make run
   ```
   *(This executes `uv run python -m src` using the default input files in `data/input/`)[cite: 1].*

3. **Run Code Quality Linters:**
   ```bash
   make lint
   ```
   *(Executes strict `flake8` and `mypy` typing checks)[cite: 1].*

## Resources
*   **Hugging Face Transformers Documentation:** Used for understanding causal LM generation and raw logit extraction.
*   **Pydantic Documentation:** Used for implementing strict data validation structs.
*   **AI Usage:** AI was utilized heavily as a sparring partner during development to conceptualize the negative infinity mask, debug PyTorch tensor formatting issues, troubleshoot the tokenizer's "space drift" behavior, and architect the dynamic wildcard state machine.

## Algorithm Explanation
The engine operates on a mathematically constrained decoding pipeline[cite: 1]. The architecture is divided into three components:
1.  **The Rulebook (State Machine):** A Finite State Machine (FSM) tracks the exact required JSON schema syntax character-by-character.
2.  **The Scanner:** It scans the model's entire 150k+ vocabulary and cross-references it with the Rulebook, returning only the Token IDs that perfectly match the required next string.
3.  **The Interceptor:** Before the model selects its next token, the engine intercepts the raw probability scores (`logits`). It applies a Negative Infinity mask (`-np.inf`) to every invalid token. The model is mathematically forced to select from the remaining structurally valid tokens.

## Design Decisions
*   **Dynamic Pydantic Structs:** Instead of hardcoding JSON logic, the program parses the `functions_definition.json` into strict Pydantic structs[cite: 1]. The FSM dynamically reads these structs to enforce the correct argument keys (e.g., forcing `"a":` and `"b":` if `fn_add_numbers` is selected)[cite: 1].
*   **Wildcard Mode vs. Exact Match:** The FSM uses exact string matching for JSON brackets and keys. However, for actual argument values, it shifts into "Wildcard Mode," allowing any digit or string token while actively monitoring for closing characters (`"` or `,`).
*   **Context Injection:** The mathematical engine is fed the function descriptions as a prefix to the raw prompt. This gives the neural network the semantic context required to calculate accurate probabilities for the argument values rather than blindly hallucinating numbers.

## Performance Analysis
*   **Reliability:** The engine achieves 100% valid JSON output[cite: 1]. Because invalid structural tokens are mathematically blocked before generation, it is physically impossible for the model to output un-parseable JSON or missing keys[cite: 1].
*   **Speed:** Utilizing the lightweight 0.6B parameter model, inference runs efficiently on local hardware[cite: 1].
*   **Accuracy:** Function selection and argument extraction consistently hit near-perfect accuracy due to the injected prompt context[cite: 1].

## Challenges Faced
1.  **The Tokenizer "Space Drift":** Tokenizers use a special character to represent preceding spaces. Initially, the engine stripped spaces for comparison, which caused the FSM to silently accept partial string matches. This caused the state machine to desync and eventually crash. The solution was implementing mathematically exact prefix matching.
2.  **Merged-Token Escape Hatches:** During wildcard generation (e.g., outputting a number), the AI would occasionally select a token that merged the value and the stop-character together (e.g., `"2,"`). The interceptor would catch the comma and drop the token, erasing the number from the final JSON. The scanner logic had to be patched to explicitly forbid merged tokens.
3.  **String Quotes:** The interceptor was originally programmed to halt wildcard generation at the first sight of a quote (`"`). This instantly crashed string generation. The FSM was refactored to dynamically handle the insertion of opening and closing quotes directly.

## Testing Strategy
The implementation was validated using a multi-tiered approach:
*   **Linters:** Ensuring strict typing and PEP-8 compliance using `mypy` and `flake8`[cite: 1].
*   **Type Validation Tests:** Testing the robustness of the Pydantic models against malformed or missing input files[cite: 1].
*   **Edge Case Prompts:** Testing prompts that require single integers, multi-digit integers, and string values (e.g., `fn_reverse_string`) to ensure the dynamic FSM and interceptor properly handled varying wildcard end-states[cite: 1].

## Example Usage
**Command:**
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

**Input Prompt:**
`"What is the sum of 265 and 345?"`

**Engine Console Output:**
```text
Processing prompt: 'What is the sum of 265 and 345?'
Raw engine output: {"name":"fn_add_numbers","parameters":{"a":265,"b":345}}
```