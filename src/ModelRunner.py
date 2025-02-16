import torch
from transformers import AutoTokenizer
from peft import PeftModel
import pandas as pd

def generate_solution(model, tokenizer, problem: str) -> str:
    instruction = "You are a coding assistant. Given the following coding problem, provide a clear and detailed solution.\n"
    input_text = instruction + problem
    print(f"Generating solution...")
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    generated_ids = model.generate(
        input_ids,
        max_length=20000,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id
    )
    print("Ids generated...")
    solution = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Decoded solution...")
    return solution

def load_model_and_tokenizer(model_path, weights_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    tokenizer = AutoTokenizer.from_pretrained("JetBrains/deepseek-coder-1.3B-kexer")
    model = torch.load(f=weights_path, map_location=device, weights_only=False)
    peft_model = PeftModel.from_pretrained(model, model_path)
    peft_model.to(device)
    print("Model initialized...")
    return (peft_model, tokenizer)
    
def runModel(
    model_path: str,
    weights_path: str,
    input_dataset_path: str,
    output_dataset_path: str
):
    data = pd.read_csv(input_dataset_path)
    try:
        for i in range(0, data.shape[0]):
            model, tokenizer = load_model_and_tokenizer(model_path, weights_path)
            column_name = "generated_solution"
            print(f"Generating solution for problem {i}...")
            solution = generate_solution(model, tokenizer, data['problem'][i])
            data.loc[data.index[i], column_name]= solution
            print(f"Generated column: {column_name}")
    except KeyboardInterrupt as e:
        print("Interrupted by user")
    finally:
        print(data.head())
        data.to_csv(output_dataset_path, index=False)