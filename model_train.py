from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import pandas as pd
import torch
from pathlib import Path

# Assets
ASSETS_DIR = Path("./assets")
dataset_path = ASSETS_DIR / "fine-tuning-small.csv" 
eval_dataset_path = ASSETS_DIR / "evaluation-small.csv"
eval_data_output_path = ASSETS_DIR / "evaluation-small-output.csv"

# Temp
OUTPUT_DIR =  "./output/deepseek_coder_v2"

# Models
MODEL_DIR = Path("./models")
WEIGHTS_PATH = MODEL_DIR / 'model_weights_ast.pth'
MODEL_PATH= MODEL_DIR / 'model_peft'

# Load pre-trained model and tokenizer
data = pd.read_csv(dataset_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'JetBrains/deepseek-coder-1.3B-kexer'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
print("Model initialized..")

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", 'gate_proj', 'up_proj', 'down_proj'],
    r=8,
    lora_alpha=64,
    lora_dropout=0.1
)
print("Memory allocated:", torch.cuda.memory_allocated() / (1024 * 1024))

def tokenize_function(example):
    inputs = tokenizer(text=example['instruction'] + example['problem'], padding="max_length", max_length=384, truncation=True)
    response = tokenizer(text=example["solution"], padding="max_length", max_length=384, truncation=True)
    
    input_ids = inputs['input_ids'] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"] + response["attention_mask"] + [1]
    label = [-100] * len(inputs['input_ids']) + response["input_ids"] + [tokenizer.pad_token_id]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': label
    }

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
print("PEFT Model Created")

data['instruction'] = "You are a coding assistant. Given the following coding problem, provide a clear and detailed solution."
data.dropna(inplace=True)
train_data = Dataset.from_pandas(data.iloc[0:int(0.8 * len(data)), :])
eval_dataset = Dataset.from_pandas(data.iloc[int(0.8 * len(data)):, :])

train_data = train_data.map(tokenize_function)
eval_dataset = eval_dataset.map(tokenize_function)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    logging_steps=10,
    num_train_epochs=10,
    save_steps=100,
    learning_rate=1e-5,
    report_to="none",
    eval_strategy="steps",
    eval_steps=100,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
)

trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_data,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)
trainer.train()

torch.save(peft_model, WEIGHTS_PATH)
peft_model.save_pretrained(MODEL_PATH)
print("Model Saved Successfully")