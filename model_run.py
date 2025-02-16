import torch
from transformers import AutoTokenizer
from peft import PeftModel
from pathlib import Path
import pandas as pd

MODEL_DIR = Path("/models")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("JetBrains/deepseek-coder-1.3B-kexer")
model = torch.load(MODEL_DIR / 'model_weights_ast.pth', map_location="cuda" if torch.cuda.is_available() else "cpu")
peft_model = PeftModel.from_pretrained(model, MODEL_DIR / 'model_peft')
peft_model.to("cuda" if torch.cuda.is_available() else "cpu")

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


print("Generating solutions for each problem...")
data = pd.read_csv('/assets/evaluation.csv')
# Generate 101 solutions for each problem
for i in range(1, 101):
    column_name = f"solution_{i}"
    data[column_name] = data['problem'].apply(generate_solution)
    print(f"Generated column: {column_name}")

# Save the updated dataset with generated solutions
data.to_csv(MODEL_DIR / 'dataset1_generated_solutions.csv', index=False)

print("Solutions saved to dataset1_generated_solutions.csv")
