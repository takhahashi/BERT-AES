import os
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from utils.utils_data import TrainDataModule
from utils.dataset import get_score_range, get_Dataset
from utils.cfunctions import simple_collate_fn
from utils.utils_models import create_module
from models.functions import return_predresults
from utils.cfunctions import regvarloss, EarlyStopping
from models.models import Scaler
from transformers import AutoModel

class Bert_reg(nn.Module):
    def __init__(self, model_name_or_path, lr):
        super(Bert_reg, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        self.lr = lr

        nn.init.normal_(self.linear.weight, std=0.02)  # 重みの初期化

    def forward(self, dataset):
        outputs = self.bert(dataset['input_ids'], token_type_ids=dataset['token_type_ids'], attention_mask=dataset['attention_mask'])
        last_hidden_state = outputs['last_hidden_state'][:, 0, :]
        score = self.sigmoid(self.linear(last_hidden_state))
        return {'score': score}


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/BERT-AES/configs", config_name="normal_reg_train")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    train_dataset = get_Dataset(cfg.model.reg_or_class,
                                cfg.path.traindata_file_name,
                                cfg.aes.prompt_id,
                                tokenizer,
                                )
    dev_dataset = get_Dataset(cfg.model.reg_or_class,
                            cfg.path.valdata_file_name,
                            cfg.aes.prompt_id,
                            tokenizer,
                            )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=cfg.training.batch_size,
                                                    shuffle=True,
                                                    collate_fn=simple_collate_fn,
                                                    )
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,
                                                    batch_size=16,
                                                    shuffle=True,
                                                    collate_fn=simple_collate_fn,
                                                    )

    model = Bert_reg(
        cfg.model.model_name_or_path,
        cfg.training.learning_rate,
        )
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    

    model.train()
    model = model.cuda()
    earlystopping = EarlyStopping(patience=cfg.training.patience, verbose=True, path=cfg.path.save_path)

    scaler = torch.cuda.amp.GradScaler()

    num_train_batch = len(train_dataloader)
    num_dev_batch = len(dev_dataloader)
    mseloss = nn.MSELoss()
    for epoch in range(cfg.training.n_epochs):
        train_loss_all = 0
        dev_loss_all = 0
        model.train()
        for idx, t_batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in t_batch.items()}
            with torch.cuda.amp.autocast():
                outputs = model(batch)
                loss = mseloss(outputs['score'].squeeze(), batch['labels'])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

            train_loss_all += loss.to('cpu').detach().numpy().copy()

        ###calibrate_step###calibrate_step###
        model.eval()
        for idx, d_batch in enumerate(dev_dataloader):
            batch = {k: v.cuda() for k, v in d_batch.items()}
            dev_score = model(batch)['score'].to('cpu').detach().numpy().copy()
            print(batch['labels'].to('cpu').detach().numpy().copy())
            dev_loss = mseloss(dev_score.squeeze(), batch['labels'].to('cpu').detach().numpy().copy())
            dev_loss_all += dev_loss

        print(f'Epoch:{epoch}, train_loss:{train_loss_all/num_train_batch}, dev_loss:{dev_loss_all/num_dev_batch}')
        earlystopping(dev_loss_all, model)
        if earlystopping.early_stop == True:
            break


if __name__ == "__main__":
    main()