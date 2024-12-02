import torch
from tqdm import tqdm
import os
from datasets import Dataset
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

tqdm.pandas()

#login()

# Definindo parâmetros diretamente no código
questions_per_prompt = 10  # Defina o número de perguntas por prompt aqui
version = 3  # Defina a versão do dataset aqui
model_name = 'gemma'  # Defina o modelo a ser usado aqui (mistral, qwen, qwen-0.5)
use_dora = False  # Se quiser usar o dora, altere para True
name = ''

max_seq_length = 12000
dtype = None
load_in_4bit = True

#model_hub_path = "Qwen/Qwen2-1.5B-Instruct"
#model_hub_path = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
model_hub_path = "mistralai/Mistral-7B-Instruct-v0.2"

# Configuração do bnb_config para quantização
bnb_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Carregar o tokenizer e o modelo
tokenizer = AutoTokenizer.from_pretrained(model_hub_path)
model = AutoModelForCausalLM.from_pretrained(model_hub_path, quantization_config=bnb_config)

# Configuração de LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
)

# Aplicar LoRA ao modelo
model = get_peft_model(model, lora_config)

train_data_path = "dataHeber/train_aviacao_qa_23q1.json"

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
df = pd.read_json(train_data_path)
#df = df[0:5]
df['text'] = df['text'] + EOS_TOKEN
dataset = Dataset.from_pandas(df)

response_template = '###Respostas:\n'

from trl import DataCollatorForCompletionOnlyLM
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Definindo argumentos de treinamento
train_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = torch.cuda.is_bfloat16_supported(),
        bf16 = torch.cuda.is_bfloat16_supported(),
        logging_steps = 1,
        save_steps=250,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "models",
        report_to="wandb",
        run_name="Gemma",
    )

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
    args = train_args
)

trainer.train()

save_path = "teste/"

# Salva o modelo e o tokenizer
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("\n\n****TRAINING COMPLETE****")
print(f"MODEL SAVE PATH: {save_path}")