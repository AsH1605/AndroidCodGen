import os
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
from model import load_model

# Load dataset
dataset_path = "/home/ash/CodeGen/fine_tuning_dataset"
dataset = load_from_disk(dataset_path)

# Load pre-trained model and tokenizer
model_name = "JetBrains/deepseek-coder-1.3B-kexer"
tokenizer, model = load_model(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=2,
    fp16=True,
    report_to="wandb",  # Log to Weights and Biases
    logging_steps=10,
    push_to_hub=False
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
