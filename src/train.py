import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from config import MODEL_NAME, DATASET_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_SEQ_LENGTH, GRADIENT_ACCUMULATION_STEPS, MODEL_SAVE_PATH, OUTPUT_DIR, LOGS_DIR
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

#charger dataset
#tokenize modelname
#bitsandbytes mistral
#PEFT pour rajouter les matrices supplémentaires à mistral
#tokenizer pour que notre model comprend sinon il s'en sortira pas
#SFTTrainer pour que entrainer notre model 

dataset = load_dataset("json", data_files = "data/train.json")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config = bnb_config, device_map = "auto", trust_remote_code = True )


lora_config = LoraConfig(
    r = LORA_R,
    lora_alpha = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    target_modules = LORA_TARGET_MODULES, 
    bias = "none",
    task_type ="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)



training_config = SFTConfig(output_dir=OUTPUT_DIR,
num_train_epochs=NUM_EPOCHS,
per_device_train_batch_size=BATCH_SIZE,
learning_rate=LEARNING_RATE,
gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
logging_dir=LOGS_DIR,
save_strategy="epoch",
max_seq_length = MAX_SEQ_LENGTH)

def format_prompt(example):
    return [f"""### Instruction:
{example['instruction']}

### Question:
{example['input']}

### Réponse:
{example['output']}"""]

trainer = SFTTrainer(model = model,
                     train_dataset = dataset["train"],
                     args = training_config,
                     processing_class = tokenizer,
                     formatting_func=format_prompt)

trainer.train()

model.save_pretrained(MODEL_SAVE_PATH)

tokenizer.save_pretrained(MODEL_SAVE_PATH)