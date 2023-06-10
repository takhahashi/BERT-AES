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
from utils.dataset import get_score_range
from utils.cfunctions import simple_collate_fn
from utils.utils_models import create_module

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
    print(cfg.wandb.project_name)
    print(cfg.wandb.project)

if __name__ == "__main__":
    main()