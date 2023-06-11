import os

import json
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
from utils.cfunctions import simple_collate_fn
from utils.utils_models import create_module
from utils.dataset import get_Dataset
from models.functions import return_predresults
from ue4nlp.ue_estimater_ensemble import UeEstimatorEnsemble
from ue4nlp.ue_estimater_trust import UeEstimatorTrustscore
from ue4nlp.ue_estimater_mcd import UeEstimatorDp

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


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/test1/configs", config_name="eval_config")
def main(cfg: DictConfig):
    test_dataset = get_Dataset(cfg.model.reg_or_class, 
                               cfg.path.testdata_file_name, 
                               cfg.aes.prompt_id, 
                               AutoTokenizer.from_pretrained(cfg.model.model_name_or_path),
                               )
    train_dataset = get_Dataset(cfg.model.reg_or_class, 
                               cfg.path.traindata_file_name, 
                               cfg.aes.prompt_id, 
                               AutoTokenizer.from_pretrained(cfg.model.model_name_or_path),
                               )
    print('---------------')
    modelpaths = cfg.ue.ensemble_model_paths
    print(type(modelpaths))
    print(cfg.ue.ensemble_model_paths)
    print('-------------------')
    if cfg.eval.collate_fn == True:
        collate_fn = simple_collate_fn
    else:
        collate_fn = None
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg.eval.batch_size,
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  )
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=cfg.eval.batch_size,
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  )
    
    
    model = create_module(cfg.model.model_name_or_path, 
                          cfg.model.reg_or_class, 
                          learning_rate=1e-5, 
                          num_labels=cfg.model.num_labels, 
                          save_path=cfg.path.model_save_path)
    
    
    eval_results = return_predresults(model, test_dataloader, rt_clsvec=False, dropout = False)

    trust_estimater = UeEstimatorTrustscore(model, 
                                            train_dataloader, 
                                            cfg.aes.prompt_id,
                                            )
    trust_estimater.fit_ue()
    trust_results = trust_estimater(test_dataloader)
    eval_results.update(trust_results)

    mcdp_estimater = UeEstimatorDp(model, 
                                   cfg.ue.num_dropout, 
                                   cfg.aes.prompt_id, 
                                   cfg.model.reg_or_class,
                                   )
    mcdp_results = mcdp_estimater(test_dataloader)
    eval_results.update(mcdp_results)

    ensemble_estimater = UeEstimatorEnsemble(cfg.ue.ensemble_model_paths,
                                             cfg.aes.prompt_id,
                                             cfg.model.reg_or_class,
                                             )
    ensemble_results = ensemble_estimater(test_dataloader)
    eval_results.update(ensemble_results)

    list_results = {k: v.tolist() for k, v in eval_results.items() if type(v) == type(np.array([1, 2, 3.]))}
    
    with open(cfg.path.results_save_path, mode="wt", encoding="utf-8") as f:
        json.dump(list_results, f, ensure_ascii=False)
    

if __name__ == "__main__":
    main()