import gc
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
    
    # if output dataset file is missing, create it
    try:
        output_df = pd.read_csv(output_dataset_path)
    except FileNotFoundError:
        print(f"Output dataset file not found. Creating a new file at {output_dataset_path}")
        output_df = pd.DataFrame(columns=["problem", "solution", "generated_solution"])
        output_df.to_csv(output_dataset_path, index=False)

    # Load the input and output files
    input_df = pd.read_csv(input_dataset_path)
    output_df = pd.read_csv(output_dataset_path)
    column_name = "generated_solution"

    try:
        for i in range(input_df.shape[0]):
            # if generated_solution is already present in output dataframe, skip
            try:
                if not pd.isnull(output_df.loc[output_df.index[i], column_name]):
                        print(f"Some value already present in column {column_name} for problem {i}. Skipping...")
                        continue
            except KeyError:
                pass
            except IndexError:
                pass

            # Else generate solution
            problem = input_df['problem'][i]
            solution = input_df['solution'][i]

            print(f"Loading model for problem {i}...")
            model, tokenizer = load_model_and_tokenizer(model_path, weights_path)
            print(f"Generating solution for problem {i}...")
            generated_solution = generate_solution(model, tokenizer, problem)

            # Write the generated solution to the output dataframe: problem, solution, generated_solution
            output_df.loc[i] = [problem, solution, generated_solution]
            
            print(f"Generated column: {column_name}")
            del model
            del tokenizer
            torch.cuda.empty_cache()  
            gc.collect()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print(output_df.head())
        output_df.to_csv(output_dataset_path, index=False)