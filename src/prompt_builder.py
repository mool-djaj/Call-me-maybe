import json
from .json_to_token import FunctionDefinition, Prompt


def build_prompt(
    functions: list[FunctionDefinition], prompt: Prompt
) -> str:
    """Choose by the requested operation, not merely shared input types."""
    catalog: list[str] = []
    has_fallback = False

    for function in functions:
        if function.name == "fn_unknown":
            has_fallback = True
            continue

        parameters = ", ".join(
            f"{name}: {definition.type}"
            for name, definition in function.parameters.items()
        ) or "none"

        catalog.append(
            f"{function.name}\n"
            f"  Action: {function.description}\n"
            f"  Inputs: {parameters}\n"
            f"  Result type: {function.returns.type}"
        )

    if has_fallback:
        catalog.append(
            "fn_unknown\n"
            "  Action: No listed function can perform the requested action."
        )

    instructions = (
        "You are a precise function selector. Do not execute functions.\n"
        "Identify the action and the result the user wants.\n"
        "Compare this action to what each function actually does.\n"
        "Different words for the same action are a match.\n"
        "Related subjects, similar inputs, or matching parameter types "
        "are NOT enough.\n"
        "Never replace the requested action with a different action.\n"
    )

    if has_fallback:
        instructions += (
            "Choose a real function whenever it can do the full task.\n"
            "Otherwise choose fn_unknown; do not guess a related tool.\n"
        )

    catalog_text = "\n\n".join(catalog)
    return (
        f"{instructions}\n"
        "Available functions:\n"
        f"{catalog_text}\n\n"
        f"User request:\n{prompt.prompt}\n\n"
        "Select by the requested ACTION, not the input words.\n"
        "Return exactly one function name from the list, nothing else.\n"
        "Function name:"
    )


def build_parameter_prompt(
    function: FunctionDefinition,
    prompt: Prompt,
    param_name: str,
    previous: dict[str, int | float | str | bool],
) -> str:
    """Ask the model to complete one parameter using prior values."""
    signature = ", ".join(
        f"{name}: {definition.type}"
        for name, definition in function.parameters.items()
    )
    args = ", ".join(
        f"{name}={json.dumps(value, ensure_ascii=False)}"
        for name, value in previous.items()
    )
    if args:
        args += ", "

    field_type = function.parameters[param_name].type
    quote = '"' if field_type == "string" else ""
    special_instruction = ""
    if field_type == "string":
        special_instruction = (
            "Return only this parameter's JSON string content. "
            "Close the string after the value. "
            "Do not generate the next parameter.\n"
            "Preserve spelling, capitalization, and spaces when copying.\n"
        )
        if "regex" in param_name.lower() or "pattern" in param_name.lower():
            special_instruction += (
                "Produce a concise regex for only what must be replaced.\n"
                "Do not use .* to match the whole sentence unnecessarily.\n"
                "In JSON, double backslashes in regex escapes.\n"
            )
    elif field_type == "boolean":
        special_instruction = "Use only JSON true or false.\n"

    return (
        "Convert the user's request into a function call.\n"
        "Extract the argument values from the request.\n"
        "Do not calculate the function's result.\n"
        f"{special_instruction}\n"
        f"User request: {prompt.prompt}\n\n"
        "Function signature:\n"
        f"{function.name}({signature})\n\n"
        "Complete this function call by providing the value:\n"
        f"{function.name}({args}{param_name}={quote}"
    )
