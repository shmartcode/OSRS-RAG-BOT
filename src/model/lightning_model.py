import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, get_linear_schedule_with_warmup
import gc


class OSRSRetrieverModel(pl.LightningModule):
    """
    PyTorch Lightning Module for fine-tuning sentence embeddings
    using In-Batch Negatives with Cross-Entropy Loss.
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-mpnet-base-v2",
        learning_rate=2e-5,
        temperature=0.05,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.save_hyperparameters()

        self.model = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)

    def compute_loss(self, batch):
        q_enc = batch["query"]
        pos_enc = batch["positive"]
        neg_enc = batch["negative"]

        q_vecs = self(q_enc["input_ids"], q_enc["attention_mask"])
        pos_vecs = self(pos_enc["input_ids"], pos_enc["attention_mask"])
        neg_vecs = self(neg_enc["input_ids"], neg_enc["attention_mask"])

        # In-Batch Negatives candidates (2B, D)
        candidates = torch.cat([pos_vecs, neg_vecs], dim=0)

        # Cosine similarity matrix (B, 2B)
        scores = torch.matmul(q_vecs, candidates.T) / self.temperature

        labels = torch.arange(q_vecs.size(0), device=q_vecs.device)
        return F.cross_entropy(scores, labels)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        # Log validation loss per epoch
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)

        try:
            total_steps = self.trainer.estimated_stepping_batches
            if total_steps == float("inf"):
                total_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs
        except Exception:
            # Fallback if stepping_batches isn't available yet
            total_steps = 1000

        total_steps = int(total_steps)
        warmup_steps = int(total_steps * 0.1)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self):
        # Clears Python heap and GPU cache after every training epoch
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        # Clears Python heap and GPU cache after evaluation completes
        gc.collect()
        torch.cuda.empty_cache()
