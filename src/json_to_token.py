import json
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ValidationError


class TypeDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, TypeDefinition]
    returns: TypeDefinition


class Prompt(BaseModel):
    prompt: str


def load_json(path: str) -> object:
    try:
        with open(path, "r") as file:
            return json.load(file)

    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")

    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON: {path}")


def load_functions(path: str) -> list[FunctionDefinition]:
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError("Functions file must contain a JSON array")

    try:
        return [
            FunctionDefinition.model_validate(item)
            for item in data
        ]

    except ValidationError as error:
        raise ValueError(
            f"Invalid function definition: {error}"
        )


def load_prompts(path: str) -> list[Prompt]:
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError("Prompts file must contain a JSON array")

    try:
        return [
            Prompt.model_validate(item)
            for item in data
        ]

    except ValidationError as error:
        raise ValueError(
            f"Invalid prompt: {error}"
        )


def tokenize_function_names(functions: list[FunctionDefinition],model: Small_LLM_Model,) -> dict[str, list[int]]:
    function_tokens = {}

    for function in functions:
        tokens = model.encode(function.name)[0].tolist()

        function_tokens[function.name] = tokens

    return function_tokens