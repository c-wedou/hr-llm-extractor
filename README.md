HR LLM Extractor
Fine-tuning de Mistral-7B-v0.1 sur le Code du Travail français avec QLoRA et PEFT. Pipeline end-to-end : préparation des données, entraînement, inférence et évaluation ROUGE. Prochaine version : LLM-as-a-judge pour comparer avec ROUGE.
Problème résolu
Les professionnels RH et juristes ont besoin de réponses précises sur le Code du Travail français sans passer par un juriste à chaque question. Mistral-7B est choisi pour trois raisons : open source (fine-tunable contrairement à GPT-4), déployable localement sans dépendance API, et utilisable commercialement sans coût par token en production.
Architecture


    hr-llm-extractor/ ├── config.py ← paramètres centralisés : modèle, LoRA, entraînement, chemins 
                      ├── data/   ├── prepare_dataset.py ← dataset HuggingFace, formatage Alpaca, split train/test
                                  ├──train.json ← 90% des données — utilisées pour l'entraînement 
                                  └── test.json ← 10% des données — jamais vues par le modèle pendant l'entraînement
                      └── src/    ├── train.py ← chargement Mistral en 4 bits, application LoRA, entraînement, sauvegarde 
                                  ├── inference.py ← chargement du modèle fine-tuné, interface question/réponse  
                                  └── evaluate.py ← évaluation sur test.json, calcul du score ROUGE-L moyen



## Installation
```bash
git clone https://github.com/ton-username/hr-llm-extractor
cd hr-llm-extractor
pip install -r requirements.txt
```
## Lancement
```bash
# 1. Préparer les données
python -m data.prepare_dataset
# 2. Entraîner le modèle
python -m src.train
# 3. Tester une question
python -m src.inference
# 4. Évaluer le modèle
python -m src.evaluate

```

## Décisions techniques

**QLoRA** : entraîner tous les poids de Mistral-7B nécessite 28 GB 
de VRAM. La quantization 4 bits réduit à 4 GB — faisable sur un GPU 
grand public.

**LoRA** : seules les matrices d'attention sont entraînées (1% des 
paramètres). Les poids originaux sont gelés — pas de catastrophic 
forgetting.

**Format Alpaca** : standard instruction/input/output pour le 
supervised fine-tuning. Compatible avec SFTTrainer de trl.

**ROUGE-L** : mesure le chevauchement de séquences entre réponse 
générée et réponse attendue. Métrique standard pour évaluer la 
génération de texte.


## Résultats

| Modèle | Train Loss | ROUGE-L moyen | Hardware |
|--------|-----------|---------------|----------|
| facebook/opt-125m | 0.4836 | 0.09 | CPU |
| mistralai/Mistral-7B-v0.1 | 0.2346 | 0.17 | CPU |


## Dataset

[louisbrulenaudet/code-travail](https://huggingface.co/datasets/louisbrulenaudet/code-travail) — 9 774 articles du Code du Travail français. Split : 8 796 train / 978 test.
