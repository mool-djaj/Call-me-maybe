from llm_sdk import Small_LLM_Model


from .json_to_token import (
    load_functions,
    load_prompts,
    tokenize_function_names,
)
from .prompt_builder import build_prompt
from .decoder import (
    get_allowed_next_ids,
    mask_logits,
)


model = Small_LLM_Model()

functions = load_functions(
    "data/input/functions_definition.json"
)

prompts = load_prompts(
    "data/input/function_calling_tests.json"
)

function_tokens = tokenize_function_names(
    functions,
    model,
)

text = build_prompt(
    functions,
    prompts[0],
)

input_ids = model.encode(text)[0].tolist()

generated = []


text = build_prompt(
    functions,
    prompts[0],
)

input_ids = model.encode(text)[0].tolist()

generated = []

while True:
    allowed_ids = get_allowed_next_ids(
        function_tokens,
        generated,
    )

    if not allowed_ids:
        break

    if len(allowed_ids) == 1:
        generated.append(allowed_ids[0])
        continue

    logits = model.get_logits_from_input_ids(
        input_ids + generated
    )

    masked = mask_logits(
        logits,
        allowed_ids,
    )

    next_id = max(
        range(len(masked)),
        key=lambda i: masked[i],
    )

    generated.append(next_id)


print("IDs:", generated)
print("Function:", model.decode(generated))