import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_dataset import OSRSDataModule  # Adjust to match your dataset module name
from lightning_model import OSRSRetrieverModel  # Adjust to match your model class name
from pytorch_lightning.callbacks import Callback
from transformers import AutoTokenizer


def main():
    # Initialize your Data Module
    datamodule = OSRSDataModule(batch_size=8, max_length=384)

    # Initialize the Lightning Model
    model = OSRSRetrieverModel(learning_rate=2e-5)

    # Set up Model Checkpointing to save your best weights
    # checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints", filename="osrs_retriever_final", save_top_k=1, monitor="val_loss", mode="min")
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="osrs_retriever_final",
        save_top_k=1,
        monitor="train_loss",  # <-- Track training loss instead
        mode="min",
        save_last=True,  # <-- Guarantees a file is saved at the very end
    )

    # Configure the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",  # Automatically picks GPU if available, else CPU
        devices=1,
        precision="16-mixed",  # Speeds up training and saves VRAM if using a GPU
        accumulate_grad_batches=2,  # 4 items * 2 steps = effective batch size of 8
        callbacks=[checkpoint_callback, SaveEveryEpochModel()],
    )

    # Kick off training
    trainer.fit(model, datamodule=datamodule)
    # Automatically save the final Hugging Face model folder too!
    model.model.save_pretrained("./fine_tuned_osrs_embedder")
    final_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    final_tokenizer.save_pretrained("./fine_tuned_osrs_embedder")
    print("Training complete and model successfully saved!")


class SaveEveryEpochModel(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # current_epoch is 0-indexed (0 = Epoch 1, 1 = Epoch 2, 2 = Epoch 3)
        epoch_num = trainer.current_epoch + 1
        save_path = f"./fine_tuned_osrs_embedder_epoch{epoch_num}"

        print(f"\nSaving model weights for Epoch {epoch_num}...")
        pl_module.model.save_pretrained(save_path)
        # Load and save the tokenizer explicitly
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        tokenizer.save_pretrained(save_path)

        print(f"Epoch {epoch_num} saved to {save_path}!")


if __name__ == "__main__":
    main()
