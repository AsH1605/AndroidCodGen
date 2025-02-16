import modal
from pathlib import Path
import modal.gpu

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    ["torch", "transformers==4.45.2", "numpy", "pandas", "torchvision", "Pillow", "scikit-learn", "sentencepiece", "peft==0.14.0", "datasets"]
)
assets = modal.Mount.from_local_dir(
    ".",
    condition=lambda pth: not ".ipynb" in pth,
    remote_path="/assets",
)
MODEL_DIR = Path("/models")
app = modal.App('CLIP_Trainer')
volume = modal.Volume.from_name("AST_MODAL", create_if_missing=True)


@app.function(image=image, mounts=[assets], gpu=modal.gpu.A100(count=8), volumes={MODEL_DIR: volume}, timeout=86400)
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    import pandas as pd
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'JetBrains/deepseek-coder-1.3B-kexer'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print("Model initialized..")

    # Generate a solution using the fine-tuned model
    def generate_solution(example_problem: str) -> str:
        instruction = "You are a coding assistant. Given the following coding problem, provide a clear and detailed solution.\n"
        input_text = instruction + example_problem

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        generated_ids = peft_model.generate(
            input_ids, 
            max_length=20000, 
            num_beams=5, 
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )
        solution = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return solution
    
    # Load the evaluation.csv file
    data = pd.read_csv('/assets/evaluation.csv')

    # Generate 101 solutions for each problem
    for i in range(1, 101):
        column_name = f"solution_{i}"
        data[column_name] = data['problem'].apply(generate_solution)
        print(f"Generated column: {column_name}")

    # Save the updated dataset with generated solutions
    data.to_csv(MODEL_DIR / 'dataset1_generated_solutions.csv', index=False)

    print("Solutions saved to dataset1_generated_solutions.csv")