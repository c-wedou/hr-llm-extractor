from datasets import load_dataset
from config import DATASET_NAME
import os


if os.path.exists("data/train.json") and os.path.exists("data/test.json"):
    print("Dataset déjà préparé, on skip.")
else:
    dataset = load_dataset(
        DATASET_NAME,
        streaming = False
    )

    def format_example(row):
        return{
            "instruction" : "Tu es un expert juridique spécialisé dans le droit du travail français. Réponds à la question suivante en te basant sur le Code du Travail.",
            "input" : f"Que dit {row['ref']} sur {row['sectionParentTitre']} ? ",
            "output" : row["texte"],
        }


    dataset_clean = dataset["train"].filter(lambda row: row["texte"] is not None and row["texte"] != "")

    dataset_formatted = dataset_clean.map(format_example)
    split = dataset_formatted.train_test_split(test_size = 0.1, seed = 42)
    split["train"].to_json("data/train.json")
    split["test"].to_json("data/test.json")

