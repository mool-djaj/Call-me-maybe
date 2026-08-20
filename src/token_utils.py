from typing import List, Dict

def text_to_token_ids(model, text: str) -> List[int]:
    tensor_ids = model.encode(text)
    return tensor_ids[0].tolist()

def find_matching_token_ids(vocab: Dict[str, int], string_rule: str, current_buffer: str) -> List[int]:
    allowed_ids = []

    if string_rule == "__VALUE_NUMBER__":
        for token_str, token_id in vocab.items():
            token_real_text = token_str.replace('Ġ', ' ')
            if not token_real_text: continue
            if ' ' in token_real_text: continue
            
            if all(c in "0123456789.-,}" for c in token_real_text):
                # PATCH: Forbid merged tokens so the interceptor doesn't drop values
                if any(stop in token_real_text for stop in ",}") and len(token_real_text) > 1:
                    continue
                allowed_ids.append(token_id)
        return allowed_ids

    elif string_rule == "__VALUE_STRING__":
        for token_str, token_id in vocab.items():
            token_real_text = token_str.replace('Ġ', ' ')
            if not token_real_text: continue
            
            if any(c in "\n[]{}" for c in token_real_text):
                continue
            # PATCH: The stop char for strings is only the quote. Forbid merged quotes.
            if '"' in token_real_text and len(token_real_text) > 1:
                continue
            allowed_ids.append(token_id)
        return allowed_ids

    # --- EXACT PREFIX MATCHING ---
    target_text = string_rule
    if not target_text.startswith(current_buffer):
        return []
        
    remaining_text = target_text[len(current_buffer):]
    
    for token_str, token_id in vocab.items():
        token_real_text = token_str.replace('Ġ', ' ')
        if not token_real_text: continue
        
        # PATCH: Exact mathematical matching prevents ' ' from being injected
        if remaining_text.startswith(token_real_text):
            allowed_ids.append(token_id)
            
    return allowed_ids