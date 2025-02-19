import modal
from pathlib import Path
import modal.gpu
from ModelRunner import runModel

app = modal.App('CLIP_Runner')

# Create a Docker image with the required dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    ["torch", "transformers==4.45.2", "numpy", "pandas", "torchvision", "Pillow", "scikit-learn", "sentencepiece", "peft==0.14.0", "datasets"]
)

# Load ./assets at /assets in the container
assets = modal.Mount.from_local_dir(
    "./assets",
    condition=lambda pth: not ".ipynb" in pth,
    remote_path="/assets",
)

# Path for input files
ASSETS_DIR = Path("/assets")
eval_dataset_path = ASSETS_DIR / "evaluation.csv" 

# Persistent storage for trained models
MODEL_DIR = Path("/models")
model_path = MODEL_DIR / 'model_peft'
weights_path = MODEL_DIR / "model_weights_ast.pth"
output_path = MODEL_DIR / "output_full.csv"

volume = modal.Volume.from_name("AST_MODAL", create_if_missing=True)
@app.function(image=image, mounts=[assets], gpu=modal.gpu.A100(count=8), volumes={MODEL_DIR: volume}, timeout=86400)
def main():
    runModel(
        model_path=model_path,
        weights_path=weights_path,
        input_dataset_path=eval_dataset_path,
        output_dataset_path=output_path
    )