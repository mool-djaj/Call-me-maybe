from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .decoder import get_allowed_next_ids, mask_logits
from .json_to_token import FunctionDefinition, Prompt, load_functions
from .prompt_builder import build_prompt

FALLBACK_NAME = "fn_unknown"
FALLBACK_PENALTY = 2.5


def load_candidates(
    path: str,
) -> tuple[list[FunctionDefinition], dict[str, FunctionDefinition]]:
    """Load real functions and add the internal fallback for routing."""
    functions = load_functions(path)
    if any(function.name == FALLBACK_NAME for function in functions):
        raise ValueError("Reserved function name: fn_unknown")

    fallback = load_functions("src/fallback_function.json")
    if len(fallback) != 1 or fallback[0].name != FALLBACK_NAME:
        raise ValueError("Fallback file must contain only fn_unknown")
    if fallback[0].parameters:
        raise ValueError("fn_unknown must not have parameters")

    by_name = {function.name: function for function in functions}
    return functions + fallback, by_name


def adjust_fallback_score(
    scores: list[float],
    generated: list[int],
    function_tokens: dict[str, list[int]],
) -> None:
    """Reduce fallback score when real alternatives exist."""

    fallback = function_tokens.get(FALLBACK_NAME, [])
    n = len(generated)

    if len(fallback) <= n:
        return

    if fallback[:n] != generated:
        return

    other_ids = set()

    for name, ids in function_tokens.items():
        if name == FALLBACK_NAME:
            continue

        if len(ids) <= n:
            continue

        if ids[:n] != generated:
            continue

        other_ids.add(ids[n])

    if other_ids and fallback[n] not in other_ids:
        scores[fallback[n]] -= FALLBACK_PENALTY


def select_function(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    prompt: Prompt,
    function_tokens: dict[str, list[int]],
) -> str:
    """Greedily choose a valid function name with masked Qwen logits."""
    input_ids = model.encode(build_prompt(functions, prompt))[0].tolist()
    generated: list[int] = []
    ending = model.encode("\n")[0].tolist()
    if len(ending) != 1:
        raise ValueError("Cannot encode function-name ending")
    end_id = ending[0]

    while True:

        completed = []
        for name, ids in function_tokens.items():
            if ids == generated:
                completed.append(name)

        allowed = get_allowed_next_ids(function_tokens, generated)
        if completed and not allowed:
            return completed[0]
        if not allowed:
            raise ValueError("No matching function continuation")
        if completed:
            allowed.append(end_id)
        if len(allowed) == 1:
            generated.append(allowed[0])
            continue

        logits = model.get_logits_from_input_ids(input_ids + generated)
        scores = mask_logits(logits, allowed)
        adjust_fallback_score(scores, generated, function_tokens)

        next_id = max(allowed, key=lambda token_id: scores[token_id])
        if completed and next_id == end_id:
            return completed[0]
        generated.append(next_id)
