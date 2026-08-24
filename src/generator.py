import numpy as np
import json
from typing import List
from src.models import FunctionDef
from src.state_machine import JSONStateMachine
from src.token_utils import find_matching_token_ids, text_to_token_ids


def generate_constrained_json(model, prompt: str, functions: List[FunctionDef]) -> str:
    machine = JSONStateMachine(functions)
    
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    # PATCH: Provide context so the LLM doesn't blindly guess numbers
    context = "Functions:\n"
    for f in functions:
        context += f"- {f.name}: {f.description}\n"
    context += f"\nRequest: {prompt}\nCall: "
    
    input_ids = text_to_token_ids(model, context)
    generated_text = ""
    
    while machine.current_state != machine.STATE_DONE:
        logits = model.get_logits_from_input_ids(input_ids)
        allowed_strings = machine.get_allowed_strings()
        is_value_state = any(rule.startswith("__VALUE_") for rule in allowed_strings)
        
        allowed_ids = []
        for string_rule in allowed_strings:
            matches = find_matching_token_ids(vocab, string_rule, machine.current_buffer)
            allowed_ids.extend(matches)
            
        allowed_ids = list(set(allowed_ids))
        
        if len(allowed_ids) == 0:
            raise ValueError(f"Scanner found 0 valid tokens at state {machine.current_state}. Buffer: '{machine.current_buffer}'")
            
        logits_array = np.array(logits)
        mask = np.full(logits_array.shape, -np.inf)
        mask[allowed_ids] = logits_array[allowed_ids]
        
        next_token_id = int(np.argmax(mask))
        next_token_string = model.decode([next_token_id])
        
        # PATCH: Stop characters rely on the specific wildcard type
        if is_value_state:
            is_string_val = any(rule == "__VALUE_STRING__" for rule in allowed_strings)
            stop_chars = '"' if is_string_val else ",}"
            
            if any(c in next_token_string for c in stop_chars):
                machine.update_state("__DONE__")
                continue
        
        input_ids.append(next_token_id)
        generated_text += next_token_string
        machine.update_state(next_token_string)
        
        if len(generated_text) > 500:
            raise RuntimeError("Generation exceeded character limits.")

    return generated_text