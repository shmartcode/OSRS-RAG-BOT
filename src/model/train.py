import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning_dataset import OSRSDataModule
from lightning_model import OSRSRetrieverModel


def main():
    data_module = OSRSDataModule(model_name="sentence-transformers/all-mpnet-base-v2", batch_size=8)

    model = OSRSRetrieverModel(model_name="sentence-transformers/all-mpnet-base-v2", learning_rate=2e-5)

    # Save checkpoint with lowest val_loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename="best-osrs-retriever-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Stop early if val_loss doesn't improve for 2 consecutive epochs
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=2, mode="min")

    trainer = pl.Trainer(
        max_epochs=4,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
