import json
from llm_sdk import Small_LLM_Model

def load_vocabulary(model: Small_LLM_Model):
    """
    Asks the model for its vocabulary file path, then loads and returns 
    the dictionary mapping token strings -> token IDs.
    """
    # Use the exact method name from the SDK
    vocab_path = model.get_path_to_vocab_file()
    
    with open(vocab_path, "r", encoding="utf-8") as file:
        vocabulary = json.load(file)
        
    return vocabulary

def get_tokens_matching_prefix(vocabulary, target_prefix):
    """
    Scans the vocabulary and returns a list of Token IDs for strings
    that start with target_prefix.
    """
    matching_ids = []
    
    for token_string, token_id in vocabulary.items():
        if token_string.startswith(target_prefix):
            matching_ids.append(token_id)
            
    return matching_ids

# --- Test Block ---
# if __name__ == "__main__":
#     print("Initializing SDK to locate vocabulary...")
#     # We must instantiate the model to use its methods
#     my_model = Small_LLM_Model()
    
#     print("Loading vocabulary JSON into memory...")
#     # Pass the model instance to our loader
#     vocab = load_vocabulary(my_model)
#     print(f"Successfully loaded {len(vocab)} total tokens into memory.")
    
#     # Test 1: Find tokens that start with the JSON opening brace '{'
#     brace_tokens = get_tokens_matching_prefix(vocab, "{")
#     print(f"\nFound {len(brace_tokens)} tokens starting with '{{':")
#     print(f"First 5 Token IDs: {brace_tokens[:5]}")
    
#     # Test 2: Find tokens that start with 'fn_'
#     fn_tokens = get_tokens_matching_prefix(vocab, "fn_")
#     print(f"\nFound {len(fn_tokens)} tokens starting with 'fn_':")
#     print(f"First 5 Token IDs: {fn_tokens[:5]}")
    