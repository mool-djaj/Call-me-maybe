import json

from llm_sdk import Small_LLM_Model


def load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    path = model.get_path_to_vocab_file()

    with open(path, "r") as file:
        return json.load(file)