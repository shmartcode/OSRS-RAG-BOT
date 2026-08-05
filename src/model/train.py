import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_dataset import OSRSDataModule  # Adjust to match your dataset module name
from lightning_model import OSRSRetrieverModel  # Adjust to match your model class name


def main():
    # Initialize your Data Module
    datamodule = OSRSDataModule(batch_size=4, max_length=384)

    # Initialize the Lightning Model
    model = OSRSRetrieverModel(learning_rate=2e-5)

    # Set up Model Checkpointing to save your best weights
    checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints", filename="osrs_retriever_final", save_top_k=1, monitor="val_loss", mode="min")

    # Configure the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",  # Automatically picks GPU if available, else CPU
        devices=1,
        precision="16-mixed",  # Speeds up training and saves VRAM if using a GPU
        accumulate_grad_batches=2,  # 4 items * 2 steps = effective batch size of 8
        callbacks=[checkpoint_callback],
    )

    # Kick off training
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
