import torch
from transformers import AutoTokenizer
from peft import PeftModel
from pathlib import Path
import pandas as pd

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

# Load tokenizer and model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")
tokenizer = AutoTokenizer.from_pretrained("JetBrains/deepseek-coder-1.3B-kexer")
model = torch.load(f=WEIGHTS_PATH, map_location=device, weights_only=False)
peft_model = PeftModel.from_pretrained(model, MODEL_PATH)
peft_model.to(device)

def generate_solution(example_problem: str) -> str:
    instruction = "You are a coding assistant. Given the following coding problem, provide a clear and detailed solution.\n"
    input_text = instruction + example_problem
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(peft_model.device)
    generated_ids = peft_model.generate(
        input_ids,
        max_length=20000,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id
    )
    solution = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return solution

def runSample():
    print("Generating solutions for each problem...")
    data = pd.read_csv(eval_dataset_path)
    for i in range(0, 1): # Run for only two rows
        column_name = f"solution_{i}"
        print(f"Generating column: {column_name}")
        data[column_name] = data['problem'].apply(generate_solution)
        print(f"Generated column: {column_name}")
    print(data.head())

def runFull():
    print("Generating solutions for each problem...")
    data = pd.read_csv(eval_dataset_path)
    for i in range(0, data.shape[0]):
        column_name = f"solution_{i}"
        print(f"Generating column: {column_name}")
        data[column_name] = data['problem'].apply(generate_solution)
        print(f"Generated column: {column_name}")
    data.to_csv(eval_data_output_path, index=False)
    print("Solutions saved to dataset1_generated_solutions.csv")

def main():
    runSample()
