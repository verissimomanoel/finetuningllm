import torch
from tqdm import tqdm
import os
from datasets import Dataset
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
tqdm.pandas()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Definindo parâmetros diretamente no código
questions_per_prompt = 10  # Defina o número de perguntas por prompt aqui
version = 3  # Defina a versão do dataset aqui
model_name = 'gemma'  # Defina o modelo a ser usado aqui (mistral, qwen, qwen-0.5)
use_dora = False  # Se quiser usar o dora, altere para True
name = ''

from huggingface_hub import login
login()


max_seq_length = 12000
dtype = None
load_in_4bit = True

#model_hub_path = "Qwen/Qwen2-0.5B-Instruct"
model_hub_path = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
#model_hub_path = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
#model_hub_path = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
from transformers import AutoModelForCausalLM, AutoTokenizer


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#model_hub_path = "unsloth/Phi-3.5-mini-instruct"

#model_hub_path = "Qwen/Qwen2-1.5B-Instruct"
##model_hub_path = "ministral/Ministral-3b-instruct"
#model_hub_path = "google/gemma-2-2b-it"
#model_hub_path = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"

#tokenizer = AutoTokenizer.from_pretrained(model_hub_path)
#model = AutoModelForCausalLM.from_pretrained(model_hub_path, quantization_config=bnb_config)


# Verifica se o CUDA está disponível
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model_hub_path = model


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_hub_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
   load_in_4bit=load_in_4bit
)

'''
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_hub_path,
    max_seq_length=max_seq_length,
    dtype=dtype
)'''



model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
)


train_data_path = "dataHeber/train_aviacao_qa_23q1.json"



EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
df = pd.read_json(train_data_path)
#df = df[0:5]
df['text'] = df['text'] + EOS_TOKEN
dataset = Dataset.from_pandas(df)

#dataset = dataset[0:5]

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
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
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

save_path = "Qwen2.5-1.5BI/"


# Salva o modelo e o tokenizer
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("\n\n****TRAINING COMPLETE****")
print(f"MODEL SAVE PATH: {save_path}")
