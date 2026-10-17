import heapq
import json
import math
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .data_type_validator import (
    get_allowed_number_ids,
    validate_number,
    validate_integer,
    validate_string_content,
)


def get_allowed_next_ids(
    function_tokens: dict[str, list[int]],
    generated: list[int],
) -> list[int]:
    """Get valid next function-name tokens for the current prefix."""
    allowed = []
    for tokens in function_tokens.values():
        if tokens[:len(generated)] == generated:
            if len(tokens) > len(generated):
                allowed.append(tokens[len(generated)])
    return sorted(set(allowed))


def mask_logits(
    logits: list[float], allowed_ids: list[int]
) -> list[float]:
    """Set all disallowed scores to negative infinity."""
    masked = [float("-inf")] * len(logits)
    for token_id in allowed_ids:
        masked[token_id] = logits[token_id]
    return masked


def decode_number(
    model: Small_LLM_Model,
    prompt: str,
    vocab: dict[str, int],
    end_char: str,
    integer_only: bool = False,
) -> int | float:
    """Generate a complete JSON number with constrained decoding."""
    input_ids = model.encode(prompt)[0].tolist()
    end_ids = model.encode(end_char)[0].tolist()

    if len(end_ids) != 1:
        raise ValueError("Ending character must be one token")

    end_id = end_ids[0]
    generated: list[int] = []

    validate = validate_integer if integer_only else validate_number

    for _ in range(40):
        current_text = model.decode(generated)
        allowed_ids = get_allowed_number_ids(vocab, current_text)
        allowed_ids = [
            token_id
            for token_id in allowed_ids
            if validate(
                current_text + model.decode([token_id])
            ) != "invalid"
        ]

        if validate(current_text) == "complete":
            allowed_ids.append(end_id)

        if not allowed_ids:
            raise ValueError("No valid number tokens")

        logits = model.get_logits_from_input_ids(input_ids + generated)
        masked = mask_logits(logits, allowed_ids)
        next_id = max(allowed_ids, key=lambda token_id: masked[token_id])

        if next_id == end_id:
            if any(char in current_text for char in ".eE"):
                value = float(current_text)
                if not math.isfinite(value):
                    raise ValueError("Number is not finite")
                return value
            return int(current_text)

        generated.append(next_id)
    raise ValueError("Number generation exceeded 40 tokens")


def completed_json_string(text: str) -> str | None:
    """Read a closed JSON string, including tokens ending with `",`."""
    try:
        value, end = json.JSONDecoder().raw_decode('"' + text)
    except json.JSONDecodeError:
        return None

    # The opening quote was supplied in the prompt, not generated.
    rest = text[end - 1:].lstrip()
    if rest and rest[0] not in ',)}]':
        return None
    return value if isinstance(value, str) else None


def decode_string(
    model: Small_LLM_Model,
    prompt: str,
    vocab: dict[str, int],
) -> str:
    """Greedily generate valid JSON string content up to a closing quote.

    The tokenizer may include a comma or closing parenthesis in the same
    token as the final quote. That token still finishes the string.
    """
    input_ids = model.encode(prompt)[0].tolist()
    known_token_ids = set(vocab.values())
    generated: list[int] = []

    for _ in range(64):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        current_text = model.decode(generated)

        # The top candidate is usually valid. Do not decode all 150k tokens.
        ranked_ids = heapq.nlargest(
            256, range(len(logits)), key=logits.__getitem__
        )

        for token_id in ranked_ids:
            if token_id not in known_token_ids:
                continue
            candidate_text = model.decode(generated + [token_id])
            if candidate_text == current_text:
                continue

            # A token such as '", regex=' contains an ending quote plus
            # additional function-call text: stop at the quote.
            finished = completed_json_string(candidate_text)
            if finished is not None:
                return finished

            if validate_string_content(candidate_text) == 'invalid':
                continue

            generated.append(token_id)
            break
        else:
            # Rare fallback when no top-256 candidate is usable.
            all_ids = sorted(
                range(len(logits)), key=logits.__getitem__, reverse=True
            )
            for token_id in all_ids[256:]:
                if token_id not in known_token_ids:
                    continue
                candidate_text = model.decode(generated + [token_id])
                finished = completed_json_string(candidate_text)
                if finished is not None:
                    return finished
                if (
                    candidate_text != current_text
                    and validate_string_content(candidate_text) != 'invalid'
                ):
                    generated.append(token_id)
                    break
            else:
                raise ValueError('No valid string continuation')

    raise ValueError(
        'String generation exceeded 64 tokens: '
        f'{model.decode(generated)[-160:]!r}'
    )


def decode_boolean(model: Small_LLM_Model, prompt: str) -> bool:
    """Use Qwen logits to choose between JSON true and false."""
    input_ids = model.encode(prompt)[0].tolist()
    choices = {
        "true": model.encode("true")[0].tolist(),
        "false": model.encode("false")[0].tolist(),
    }
    if any(not ids for ids in choices.values()):
        raise ValueError("Boolean tokenization failed")

    generated: list[int] = []
    for _ in range(12):
        for word, ids in choices.items():
            if generated == ids:
                return word == "true"

        allowed = get_allowed_next_ids(choices, generated)
        if not allowed:
            raise ValueError("No valid boolean token continuation")

        if len(allowed) == 1:
            generated.append(allowed[0])
            continue

        logits = model.get_logits_from_input_ids(input_ids + generated)
        # Greedy choice among valid tokens equals masking then argmax.
        generated.append(max(allowed, key=lambda token: logits[token]))

    raise ValueError("Boolean generation exceeded 12 tokens")
