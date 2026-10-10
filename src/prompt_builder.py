"""Build contexts for model-guided function and parameter extraction."""

import json

from .json_to_token import FunctionDefinition, Prompt


def build_prompt(
    functions: list[FunctionDefinition], prompt: Prompt
) -> str:
    """Present candidate functions and request one function name."""
    definitions = [function.model_dump() for function in functions]
    return (
        "Available functions:\n"
        f"{json.dumps(definitions, ensure_ascii=False, indent=2)}\n\n"
        f"User request:\n{prompt.prompt}\n\n"
        "Choose the best function.\nFunction name:"
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
            "Preserve the spelling and letter case of quoted or named "
            "text in the request.\n"
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
