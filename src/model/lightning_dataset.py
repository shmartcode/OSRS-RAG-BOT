import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

PROCESSED_DIR = "data/processed"


class OSRSTripletDataset(Dataset):
    """
    Custom PyTorch Dataset that loads generated OSRS training pairs.
    """

    def __init__(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negative": item["negative"],
        }


class OSRSDataModule(pl.LightningDataModule):

    def __init__(
        self,
        model_name="sentence-transformers/all-mpnet-base-v2",
        batch_size=8,
        max_query_length=64,
        max_passage_length=384,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage=None):
        train_path = os.path.join(PROCESSED_DIR, "train_pairs.json")
        val_path = os.path.join(PROCESSED_DIR, "val_pairs.json")

        self.train_dataset = OSRSTripletDataset(train_path)
        self.val_dataset = OSRSTripletDataset(val_path)

    def collate_fn(self, batch):
        queries = [item["query"] for item in batch]
        positives = [item["positive"] for item in batch]
        negatives = [item["negative"] for item in batch]

        query_enc = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        pos_enc = self.tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=self.max_passage_length,
            return_tensors="pt",
        )
        neg_enc = self.tokenizer(
            negatives,
            padding=True,
            truncation=True,
            max_length=self.max_passage_length,
            return_tensors="pt",
        )

        return {"query": query_enc, "positive": pos_enc, "negative": neg_enc}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle validation data
            collate_fn=self.collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
