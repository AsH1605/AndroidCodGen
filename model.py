from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name="JetBrains/deepseek-coder-1.3B-kexer"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    return tokenizer, model
