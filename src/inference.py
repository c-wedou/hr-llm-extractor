
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_SAVE_PATH
model = AutoModelForCausalLM.from_pretrained(MODEL_SAVE_PATH)

# charger le tokenizer — une seule fois au début
tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
question = input("Quelle est ta question?")
# tokenizer la question — après input()
tokens = tokenizer(question, return_tensors="pt")
output_tokens = model.generate(tokens["input_ids"], max_new_tokens=200, repetition_penalty=1.3,
    temperature=0.7,
    do_sample=True)
reponse = tokenizer.decode(output_tokens[0])      
print(reponse)