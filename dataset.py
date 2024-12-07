from datasets import load_dataset

def preprocess_dataset(input_path, output_path):
    dataset = load_dataset("csv", data_files=input_path)
    dataset = dataset.map(
        lambda x: {"input": x["problem"], "output": x["solution"]}
    )
    dataset.save_to_disk(output_path)
    print(f"Dataset preprocessed and saved to {output_path}")

if __name__ == "__main__":
    input_path = "/home/ash/CodeGen/fine-tuning.csv"
    output_path = "/home/ash/CodeGen/fine_tuning_dataset"
    preprocess_dataset(input_path, output_path)
