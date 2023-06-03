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
from transformers import AutoTokenizer
from utils.utils_data import TrainDataModule
from utils.functions import simple_collate_fn
from utils.utils_models import create_module
from utils.dataset import get_Dataset

def make_callbacks(min_delta, patience, checkpoint_path, filename):

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=filename,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=patience, mode="min"
    )

    return [early_stop_callback, checkpoint_callback]


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/test1/configs", config_name="config")
def main(cfg: DictConfig):
    test_dataset = get_Dataset(cfg.model.reg_or_class, 
                               cfg.path.testdata_file_name, 
                               cfg.aes.prompt_id, 
                               AutoTokenizer.from_pretrained(cfg.model.model_name_or_path),
                               )
    if cfg.test.collate_fn == True:
        collate_fn = simple_collate_fn
    else:
        collate_fn = None
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg.test.batch_size,
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  )
    
    model = create_module(cfg.model.model_name_or_path, 
                          cfg.model.reg_or_class, 
                          learning_rate=1e-5, 
                          num_labels=cfg.model.num_labels, 
                          save_path=cfg.path.model_save_path)
    
    eval_results = {}
    if cfg.ue.detar
        for t_data in test_dataloader:
            batch = {k: v.cuda() for k, v in t_data.items()}
            y_true = {'labels': batch['labels'].to('cpu').detach().numpy().copy()}
            x = {'input_ids':batch['input_ids'],
                        'attention_mask':batch['attention_mask'],
                        'token_type_ids':batch['token_type_ids']}
            outputs = {k: v.to('cpu').detach().numpy().copy() for k, v in model(x).items()}
            if len(eval_results) == 0:
                eval_results.update(y_true)
                eval_results.update({k: v.flatten() for k, v in outputs.items()})
            else:
                y_true.update(outputs)
                eval_results = {k1: np.concatenate([v1, v2.flatten()]) for (k1, v1), (k2, v2) in zip(eval_results.items(), y_true.items())}
    
    dropout, maha, trustscore, 


    cwd = hydra.utils.get_original_cwd()
    wandb_logger = WandbLogger(
        name=(cfg.wandb.project_name),
        project=cfg.wandb.project,
        tags=cfg.wandb.tags,
        log_model=True,
    )
    checkpoint_path = cfg.path.checkpoint_path
    wandb_logger.log_hyperparams(cfg)
    if cfg.training.collate_fn:
        collate_fn = simple_collate_fn
    else:
        collate_fn = None
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    data_module = TrainDataModule(cfg.model.reg_or_class, 
                                  cfg.path.traindata_file_name,
                                  cfg.path.valdata_file_name,
                                  tokenizer,
                                  cfg.training.batch_size,
                                  cfg.model.max_length,
                                  cfg.aes.prompt_id,
                                  collate_fn=collate_fn)

    data_module.setup()

    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, checkpoint_path, cfg.path.save_filename,
    )
    model = create_module(cfg.model.model_name_or_path, cfg.model.reg_or_class, cfg.training.learning_rate, num_labels=cfg.model.num_labels)
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        callbacks=call_backs, 
        logger=wandb_logger,
        accelerator="gpu", 
        devices=1,
        precision=16,
    )
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()