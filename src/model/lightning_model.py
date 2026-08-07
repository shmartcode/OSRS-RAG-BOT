import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel


class OSRSRetrieverModel(pl.LightningModule):
    """
    PyTorch Lightning Module for fine-tuning the sentence embedding model
    using triplet loss (Query, Positive, Negative).
    """

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", learning_rate=2e-5):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        # Load the pre-trained transformer backbone
        self.model = AutoModel.from_pretrained(model_name)
        self.model.train()

    def mean_pooling(self, model_output, attention_mask):
        """
        Averages token embeddings across the attention mask to create
        a single fixed-size sentence/passage vector.
        """
        token_embeddings = model_output[0]  # First element contains hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        # Pass inputs through the transformer backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Apply mean pooling to get dense vector embeddings
        embeddings = self.mean_pooling(outputs, attention_mask)
        # Normalize embeddings to unit length for clean cosine similarity computation
        return F.normalize(embeddings, p=2, dim=1)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass for Queries, Positive matches, and Negative matches
        q_enc = batch["query"]
        pos_enc = batch["positive"]
        neg_enc = batch["negative"]

        q_vecs = self(q_enc["input_ids"], q_enc["attention_mask"])
        pos_vecs = self(pos_enc["input_ids"], pos_enc["attention_mask"])
        neg_vecs = self(neg_enc["input_ids"], neg_enc["attention_mask"])

        # 2. Calculate Cosine Similarities
        # Distance/similarity between query and the correct (positive) passage
        pos_score = F.cosine_similarity(q_vecs, pos_vecs)
        # Distance/similarity between query and the incorrect (negative) passage
        neg_score = F.cosine_similarity(q_vecs, neg_vecs)

        # 3. Triplet Margin Loss Optimization
        # We want pos_score to be significantly higher than neg_score (margin = 0.5)
        margin = 0.5
        loss = F.relu(neg_score - pos_score + margin).mean()

        # Log training loss for monitoring progress
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Standard AdamW optimizer configured with the learning rate
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
