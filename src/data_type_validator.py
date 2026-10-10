def validate_number(text: str) -> str:
    """Return complete, incomplete, or invalid for a JSON number prefix."""
    state = "start"

    for char in text:
        if state in ("start", "sign"):
            if char == "0":
                state = "zero"
            elif "1" <= char <= "9":
                state = "integer"
            elif state == "start" and char == "-":
                state = "sign"
            else:
                return "invalid"

        elif state == "zero":
            if char == ".":
                state = "dot"
            elif char in "eE":
                state = "exponent"
            else:
                return "invalid"

        elif state == "integer":
            if "0" <= char <= "9":
                continue
            elif char == ".":
                state = "dot"
            elif char in "eE":
                state = "exponent"
            else:
                return "invalid"

        elif state == "dot":
            if "0" <= char <= "9":
                state = "fraction"
            else:
                return "invalid"

        elif state == "fraction":
            if "0" <= char <= "9":
                continue
            elif char in "eE":
                state = "exponent"
            else:
                return "invalid"

        elif state == "exponent":
            if char in "+-":
                state = "exponent_sign"
            elif "0" <= char <= "9":
                state = "exponent_digits"
            else:
                return "invalid"

        elif state == "exponent_sign":
            if "0" <= char <= "9":
                state = "exponent_digits"
            else:
                return "invalid"

        elif state == "exponent_digits":
            if not "0" <= char <= "9":
                return "invalid"

    if state in ("zero", "integer", "fraction", "exponent_digits"):
        return "complete"
    return "incomplete"


def get_allowed_number_ids(
    vocab: dict[str, int], generated_text: str
) -> list[int]:
    """Find vocabulary entries that keep a number prefix possible."""
    allowed = []

    for token, token_id in vocab.items():
        candidate = generated_text + token
        if token and validate_number(candidate) != "invalid":
            allowed.append(token_id)

    return allowed


def validate_string_content(text: str) -> str:
    """Check a JSON string's content, excluding its opening quote.

    A final unescaped quote closes the value. Partial escape sequences
    remain incomplete; invalid control characters are rejected.
    """
    state = "text"
    unicode_digits = 0

    for index, char in enumerate(text):
        if state == "text":
            if char == '"':
                return "complete" if index == len(text) - 1 else "invalid"
            if char == "\\":
                state = "escape"
            elif ord(char) < 32:
                return "invalid"

        elif state == "escape":
            if char in '"\\/bfnrt':
                state = "text"
            elif char == "u":
                state = "unicode"
                unicode_digits = 0
            else:
                return "invalid"

        elif state == "unicode":
            if char not in "0123456789abcdefABCDEF":
                return "invalid"
            unicode_digits += 1
            if unicode_digits == 4:
                state = "text"

    return "incomplete"


def validate_integer(text: str) -> str:
    """Return the status of an integer-only JSON numeric prefix."""
    if text == "" or text == "-":
        return "incomplete"
    digits = text[1:] if text.startswith("-") else text
    if not digits or not all("0" <= c <= "9" for c in digits):
        return "invalid"
    if len(digits) > 1 and digits.startswith("0"):
        return "invalid"
    return "complete"
