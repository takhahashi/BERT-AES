import os

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy, auroc
from transformers import AutoTokenizer, AutoModel


class BertDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        prompt_id,
        tokenizer: AutoTokenizer,
        max_length: int,
        friendly_score: bool
    ):
        self.asap_ranges = {
          0: (0, 60),
          1: (2,12),
          2: (1,6),
          3: (0,3),
          4: (0,3),
          5: (0,4),
          6: (0,4),
          7: (0,30),
          8: (0,60)
        }
        self.input_ids, self.attention_mask, self.token_type_ids = self._get_tokenize_results(df, prompt_id, tokenizer, max_length)
        self.labels = self._get_score(df, prompt_id, friendly_score)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):

        return {'input_ids': self.input_ids[index], 
            'attention_mask': self.attention_mask[index], 
            'token_type_ids': self.token_type_ids[index], 
            'labels': self.labels[index]}
    
    def _get_score(self, df, prompt_id, friendly_score):
        promptdf = df[df['essay_set'] == prompt_id]
        labels = np.array(promptdf['domain1_score'])
        if friendly_score:
            low, high = self.asap_ranges[prompt_id]
            return torch.tensor((labels - low) / (high - low), dtype=torch.float32)
        else:
            return torch.tensor(labels, dtype=torch.float32)

    def _get_tokenize_results(self, df, prompt_id, tokenizer, max_length):
      promptdf = df[df['essay_set'] == prompt_id]
      x = promptdf['essay'].tolist()
      encoding = tokenizer(x, 
                           max_length=max_length, 
                           padding='max_length', 
                           truncation=True, 
                           return_tensors='pt')
      return encoding['input_ids'], encoding['attention_mask'], encoding['token_type_ids']


class CreateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        valid_df,
        batch_size,
        max_length,
        prompt_id,
        friendly_score=True,
        collate_fn=None,
        pretrained_model='bert-base-uncased',
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.max_length = max_length
        self.prompt_id = prompt_id
        self.friendly_score = friendly_score
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def setup(self, stage=None):
        self.train_dataset = BertDataset(
            self.train_df,
            self.prompt_id,
            self.tokenizer,
            self.max_length,
            self.friendly_score,
        )
        self.vaild_dataset = BertDataset(
            self.valid_df,
            self.prompt_id,
            self.tokenizer,
            self.max_length,
            self.friendly_score,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.vaild_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn,
        )


class CustumBert(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        pretrained_model='bert-base-uncased',
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.linear1 = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        self.lr = learning_rate
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        outputs = self.bert(**inputs)['last_hidden_state'][:, 0, :]
        score = self.sigmoid(self.linear1(outputs))
        return score

    def training_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        y_hat = self.forward(inputs=x)
        loss = self.criterion(y_hat, y)
        return {"loss": loss, "batch_preds": y_hat, "batch_labels": y}

    def validation_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        y_hat = self.forward(inputs=x)
        loss = self.criterion(y_hat, y)
        return {"loss": loss, "batch_preds": y_hat, "batch_labels": y}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def make_callbacks(min_delta, patience, checkpoint_path):

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch}",
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=patience, mode="min"
    )

    return [early_stop_callback, checkpoint_callback]

def simple_collate_fn(list_of_data):
  pad_max_len = torch.tensor(0)
  for data in list_of_data:
    if(torch.count_nonzero(data['attention_mask']) > pad_max_len):
      pad_max_len = torch.count_nonzero(data['attention_mask'])
  in_ids, token_type, atten_mask, labels = [], [], [], []
  for data in list_of_data:
    in_ids.append(data['input_ids'][:pad_max_len])
    token_type.append(data['token_type_ids'][:pad_max_len])
    atten_mask.append(data['attention_mask'][:pad_max_len])
    labels.append(data['labels'])
  batched_tensor = {}
  batched_tensor['input_ids'] = torch.stack(in_ids)
  batched_tensor['token_type_ids'] = torch.stack(token_type)
  batched_tensor['attention_mask'] = torch.stack(atten_mask)
  batched_tensor['labels'] = torch.tensor(labels)
  return batched_tensor

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    wandb_logger = WandbLogger(
        name=("exp_" + str(cfg.wandb.exp_num)),
        project=cfg.wandb.project,
        tags=cfg.wandb.tags,
        log_model=True,
    )
    checkpoint_path = os.path.join(
        wandb_logger.experiment.dir, cfg.path.checkpoint_path
    )
    wandb_logger.log_hyperparams(cfg)
    traindf = pd.read_table(cfg.path.traindata_file_name, sep='\t')
    devdf = pd.read_table(cfg.path.devdata_file_name, sep='\t')
    if cfg.training.collate_fn:
        collate_fn = simple_collate_fn
    else:
        collate_fn = None

    data_module = CreateDataModule(
        traindf,
        devdf,
        cfg.training.batch_size,
        cfg.model.max_length,
        cfg.aes.prompt_id,
        cfg.aes.friendly_score,
        collate_fn
    )
    data_module.setup()

    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, checkpoint_path
    )
    model = CustumBert(
        learning_rate=cfg.training.learning_rate
    )
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        callbacks=call_backs,
        logger=wandb_logger,
        accelerator="gpu", 
        gpus=[0]
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()