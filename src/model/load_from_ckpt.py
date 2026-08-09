import sys
from pathlib import Path
from transformers import AutoTokenizer

# Ensure src modules are in Python path
sys.path.extend(["src", "src/model"])

from lightning_model import OSRSRetrieverModel

ckpt_path = "checkpoints/best-osrs-retriever-epoch=05-val_loss=0.9906.ckpt"
output_dir = Path("fine_tuned_osrs_embedder_v2")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading checkpoint from: {ckpt_path}")
model_module = OSRSRetrieverModel.load_from_checkpoint(ckpt_path, map_location="cpu")

# 1. Save MPNet weights & config
print(f"Saving MPNet model weights to: {output_dir}")
model_module.model.save_pretrained(str(output_dir))

# 2. Save the matching tokenizer
if hasattr(model_module, "tokenizer") and model_module.tokenizer is not None:
    print("Saving tokenizer from LightningModule...")
    model_module.tokenizer.save_pretrained(str(output_dir))
else:
    print("Loading and saving base MPNet tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    tokenizer.save_pretrained(str(output_dir))

print("\nExport complete! Directory contents:")
for f in output_dir.iterdir():
    print(" -", f.name)
