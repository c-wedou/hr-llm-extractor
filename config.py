## config.py 


## model config 
#MODEL_NAME = "facebook/opt-125m"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

### Lora CONFIG

LORA_R = 16
LORA_ALPHA  = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj","k_proj", "v_proj", "o_proj"]


### TRAINING CONFIG
# 
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512
GRADIENT_ACCUMULATION_STEPS = 4


### PATHS


DATASET_NAME = "louisbrulenaudet/code-travail"
DATASET_SPLIT = "train"
HF_TOKEN = ""  # token HuggingFace si le modèle est gated

OUTPUT_DIR = "outputs/"
MODEL_SAVE_PATH = "outputs/model"
LOGS_DIR = "outputs/logs"