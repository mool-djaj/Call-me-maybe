def get_allowed_next_ids(
    function_tokens: dict[str, list[int]],
    generated: list[int],
) -> list[int]:
    allowed = []

    for tokens in function_tokens.values():
        if tokens[:len(generated)] == generated:
            if len(tokens) > len(generated):
                allowed.append(tokens[len(generated)])

    return list(set(allowed))