from dataset import get_Dataset
import torch
import pytorch_lightning as pl

class TrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        reg_or_class,
        train_datapath,
        valid_datapath,
        tokenizer,
        batch_size,
        max_length,
        prompt_id,
        friendly_score=True,
        collate_fn=None,
    ):
        super().__init__()
        self.reg_or_class = reg_or_class
        self.train_path = train_datapath
        self.valid_path = valid_datapath
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.max_length = max_length
        self.prompt_id = prompt_id
        self.friendly_score = friendly_score
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        self.train_dataset = get_Dataset(
            self.reg_or_class,
            self.train_path,
            self.prompt_id,
            self.tokenizer,
        )
        self.vaild_dataset = get_Dataset(
            self.reg_or_class,
            self.valid_path,
            self.prompt_id,
            self.tokenizer,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.vaild_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn,
        )