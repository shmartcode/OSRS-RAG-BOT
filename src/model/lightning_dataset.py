import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

PROCESSED_DIR = "data/processed"


class OSRSTripletDataset(Dataset):
    """Custom PyTorch Dataset that loads the generated OSRS training pairs
    (queries, positive passages, and negative passages) from disk.
    """

    def __init__(self, data_path, tokenizer, max_length=256):
        with open(data_path, "r") as f:
            self.pairs = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # Returns the total number of training pairs in the dataset
        return len(self.pairs)

    def __getitem__(self, idx):
        # Retrieves a single training dictionary item by index
        item = self.pairs[idx]
        return {"query": item["query"], "positive": item["positive"], "negative": item["negative"]}


class OSRSDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule to handle data loading, tokenization,
    and batching for the contrastive retriever model.
    """

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", batch_size=8, max_length=384):
        # 2060 Super: batch size 8. length 384. num workers 4-6
        # 4050L: batch size 4. length 256.  num workers 6-8
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        # Initialize the Hugging Face tokenizer corresponding to your base embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage=None):
        # Load the 1GB training pairs JSON file into the Dataset instance
        data_path = os.path.join(PROCESSED_DIR, "training_pairs.json")
        self.dataset = OSRSTripletDataset(data_path, self.tokenizer, self.max_length)

    def collate_fn(self, batch):
        """
        Custom collation function to group individual dataset samples into
        batched tensors ready for the transformer model.
        """
        queries = [item["query"] for item in batch]
        positives = [item["positive"] for item in batch]
        negatives = [item["negative"] for item in batch]

        # Tokenize queries, positive texts, and negative texts separately with padding and truncation
        query_enc = self.tokenizer(queries, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        pos_enc = self.tokenizer(positives, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        neg_enc = self.tokenizer(negatives, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        return {"query": query_enc, "positive": pos_enc, "negative": neg_enc}

    def train_dataloader(self):
        # Returns the DataLoader configured for the training loop
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle data every epoch to ensure better model generalization
            collate_fn=self.collate_fn,
            # workers for 2060: 4-6. for  4050 6-8
            num_workers=0,  # Number of subprocesses for data loading (adjust based on your CPU cores)
        )
