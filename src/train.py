import modal
from pathlib import Path
import modal.gpu

from ModelTrainer import trainModel

app = modal.App('CLIP_Trainer')
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
train_dataset_path = ASSETS_DIR / "fine-tuning.csv" 

# Persistent storage for trained models
MODEL_DIR = Path("/models")
volume = modal.Volume.from_name("AST_MODAL_MONO_JAVA", create_if_missing=True)

@app.function(image=image, mounts=[assets], gpu=modal.gpu.A100(count=8), volumes={MODEL_DIR: volume}, timeout=86400)
def main():
    trainModel(train_dataset_path, MODEL_DIR)