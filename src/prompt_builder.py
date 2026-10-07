import json

from json_parser import FunctionDefinition, Prompt


def build_prompt(
    functions: list[FunctionDefinition],
    prompt: Prompt,
) -> str:
    functions_data = [
        function.model_dump()
        for function in functions
    ]

    return f"""
Available functions:
{json.dumps(functions_data, indent=2)}

User request:
{prompt.prompt}

Return the function call as JSON:
"""