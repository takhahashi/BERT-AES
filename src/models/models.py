import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoModel
from utils.cfunctions import regvarloss

import wandb

class Bert(nn.Module):
    def __init__(self, model_name_or_path):
        super(Bert, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, dataset):
        outputs = self.bert(dataset['input_ids'], token_type_ids=dataset['token_type_ids'], attention_mask=dataset['attention_mask'])
        sequence_output = outputs['last_hidden_state'][:, 0, :]
        return {'hidden_state': sequence_output}


class BertReg(pl.LightningModule):
    def __init__(
        self,
        bert,
        learning_rate: float,
        criterion=regvarloss,
    ):
        super().__init__()
        self.bert = bert
        self.linear1 = nn.Linear(768, 1)
        self.linear2 = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        self.lr = learning_rate
        self.criterion = criterion

    def forward(self, inputs):
        hidden_state = self.bert(inputs)['hidden_state']
        score = self.sigmoid(self.linear1(hidden_state))
        logvar = self.linear2(hidden_state)
        return {'score': score, 'logvar': logvar}

    def training_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        outputs = {k: v for k, v in self.forward(x).items()}
        y_hat = outputs['score']
        logvar = outputs['logvar']
        loss = self.criterion(y, y_hat, logvar)
        self.log("train_loss", loss)
        wandb.log({"train_loss":loss})
        return {"loss": loss, "batch_preds": y_hat, "logvar": logvar, "batch_labels": y}

    def validation_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        outputs = {k: v for k, v in self.forward(x).items()}
        y_hat = outputs['score']
        logvar = outputs['logvar']
        loss = self.criterion(y, y_hat, logvar)
        self.log("val_loss", loss)
        wandb.log({"val_loss":loss})
        return {"loss": loss, "batch_preds": y_hat, "logvar": logvar, "batch_labels": y}
    
    def test_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        outputs = {k: v for k, v in self.forward(x).items()}
        y_hat = outputs['score']
        logvar = outputs['logvar']
        loss = self.criterion(y, y_hat, logvar)
        return {"loss": loss, "batch_preds": y_hat, "logvar": logvar, "batch_labels": y}

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
    
class BertClass(pl.LightningModule):
    def __init__(
        self,
        bert,
        num_classes,
        learning_rate: float,
        criterion=nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.bert = bert
        self.linear1 = nn.Linear(768, num_classes)
        self.lr = learning_rate
        self.criterion = criterion

    def forward(self, inputs):
        hidden_state = self.bert(inputs)['hidden_state']
        return {'logits': self.linear1(hidden_state)}

    def training_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        y_hat = torch.argmax(logits, dim=-1)
        return {"loss": loss, "batch_preds": y_hat, "batch_labels": y}

    def validation_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss)
        y_hat = torch.argmax(logits, dim=-1)
        return {"loss": loss, "batch_preds": y_hat, "batch_labels": y}

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
