from llm_sdk import Small_LLM_Model
from src.vocab import load_vocab


model = Small_LLM_Model()

vocab = load_vocab(model)

print(type(vocab))
print(len(vocab))

print(vocab.get("ASS"))