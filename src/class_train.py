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
from utils.cfunctions import simple_collate_fn, EarlyStopping
from utils.utils_models import create_module
from models.functions import return_predresults
from utils.cfunctions import regvarloss
from models.models import Scaler

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/BERT-AES/configs", config_name="class_config")
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
                                                    batch_size=cfg.training.batch_size,
                                                    shuffle=True,
                                                    collate_fn=simple_collate_fn,
                                                    )
    low, high = get_score_range(cfg.aes.prompt_id)
    num_labels = high - low + 1
    model = create_module(
        cfg.model.model_name_or_path,
        cfg.model.reg_or_class,
        cfg.training.learning_rate,
        num_labels=num_labels,
        spectral_norm=cfg.model.spectral_norm
        )
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)

    model.train()
    model = model.cuda()

    if cfg.model.spectral_norm == True:
        save_path = cfg.path.save_path + '_sepctralnorm'
        print('SpectralNorm is applyed!')
    else:
        save_path = cfg.path.save_path
        print('SpectralNorm is not applyed!')
    earlystopping = EarlyStopping(patience=cfg.training.patience, verbose=True, path=save_path)

    scaler = torch.cuda.amp.GradScaler()

    num_train_batch = len(train_dataloader)
    num_dev_batch = len(dev_dataloader)
    for epoch in range(cfg.training.n_epochs):
        train_loss_all = 0
        dev_loss_all = 0
        model.train()
        for idx, t_batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in t_batch.items()}
            if epoch == 0 and idx == 0:
                print(batch['labels'])
            with torch.cuda.amp.autocast():
                training_step_outputs = model.training_step(batch, idx)
            scaler.scale(training_step_outputs['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

            train_loss_all += training_step_outputs['loss'].to('cpu').detach().numpy().copy()

        model.eval()
        for idx, d_batch in enumerate(dev_dataloader):
            batch = {k: v.cuda() for k, v in d_batch.items()}
            dev_step_outputs = model.validation_step(batch, idx)
            dev_loss_all += dev_step_outputs['loss'].to('cpu').detach().numpy().copy()

        print(f'Epoch:{epoch}, train_loss:{train_loss_all/num_train_batch}, dev_loss:{dev_loss_all}, dev_loss_mean:{dev_loss_all/num_dev_batch}')
        earlystopping(dev_loss_all, model)
        if earlystopping.early_stop == True:
            break

if __name__ == "__main__":
    main()