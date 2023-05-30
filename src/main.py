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

def make_callbacks(min_delta, patience, checkpoint_path):

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=patience, mode="min"
    )

    return [early_stop_callback, checkpoint_callback]


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
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
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_module = TrainDataModule(cfg.model.reg_or_class, 
                                  cfg.path.traindata_file_name,
                                  cfg.paht.valdata_file_name,
                                  tokenizer,
                                  cfg.training.batch_size,
                                  cfg.model.max_lengh,
                                  cfg.aes.prompt_id,
                                  collate_fn=collate_fn)

    data_module.setup()

    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, checkpoint_path
    )
    model = create_module(cfg.model.reg_or_class, cfg.training.learning_rate, num_labels=cfg.model.num_labels)
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        callbacks=call_backs, 
        logger=wandb_logger,
        accelerator="gpu", 
        devices=1,
        precision=16,
    )
    trainer.fit(model, data_module)
"""
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    print(cwd)
    print(os.getcwd())
"""
if __name__ == "__main__":
    main()