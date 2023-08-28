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
from ue4nlp.ue_estimater_calibvar import UeEstimatorCalibvar
from ue4nlp.ue_estimater_mahalanobis import UeEstimatorMahalanobis
from utils.acc_metric_func import calc_qwk
from utils.cfunctions import simple_collate_fn, score_f2int


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/BERT-AES/configs", config_name="acc_reg_eachmodel")
def main(cfg: DictConfig):
    test_dataset = get_Dataset(cfg.model.reg_or_class, 
                               cfg.path.testdata_file_name, 
                               cfg.aes.prompt_id, 
                               AutoTokenizer.from_pretrained(cfg.model.model_name_or_path),
                               )
    
    if cfg.eval.collate_fn == True:
        collate_fn = simple_collate_fn
    else:
        collate_fn = None
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg.eval.batch_size,
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  )
    
    model = create_module(cfg.model.model_name_or_path, 
                          cfg.model.reg_or_class, 
                          learning_rate=1e-5, 
                          num_labels=cfg.model.num_labels, 
                          )
    model.load_state_dict(torch.load(cfg.path.model_save_path))


    test_results = return_predresults(model, test_dataloader, rt_clsvec=False, dropout=False)

    true = score_f2int(test_results['labels'], cfg.aes.prompt_id)
    pred = score_f2int(test_results['score'], cfg.aes.prompt_id)

    results_dic = {'qwk': np.corrcoef(true, pred)[0][1], 
                    'corr': calc_qwk(true, pred, cfg.aes.prompt_id, 'reg'), 
                    'rmse': np.sqrt((true - pred) ** 2).mean()}


    list_results = {k: v.tolist() for k, v in results_dic.items() if type(v) == type(np.array([1, 2, 3.]))}
    with open(cfg.path.results_save_path, mode="wt", encoding="utf-8") as f:
        json.dump(list_results, f, ensure_ascii=False)

if __name__ == "__main__":
    main()