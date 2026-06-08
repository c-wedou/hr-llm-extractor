from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from config import MODEL_SAVE_PATH
from rouge_score import rouge_scorer
import numpy as np


dataset = load_dataset(
    "json",
    data_files ="data/test.json"
    )

model = AutoModelForCausalLM.from_pretrained(MODEL_SAVE_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer = True)
scores_total = []
for example in dataset["train"]:
    prompt = f"""Tu es un expert juridique spécialisé dans le droit du travail français. Réponds à la question suivante en te basant sur le Code du Travail.
    ### Question :
    {example['input']}

    ### Reponse :
    """
    tokens = tokenizer(prompt, return_tensors = "pt")
    outputs_tokens = model.generate(tokens["input_ids"], max_new_tokens = 200)
    reponse = tokenizer.decode(outputs_tokens[0])
    reponse_attendue = example["output"]

    scores = scorer.score(reponse_attendue, reponse)
    rougeL = scores["rougeL"].fmeasure
    scores_total.append(rougeL)
    print(f"ROUGE-L: {rougeL:.4f}")

print(f"ROUGE-L moyen: {np.mean(scores_total):.4f}")