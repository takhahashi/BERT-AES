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
from utils.ue_metric_func import calc_rcc_auc, calc_rpp, calc_roc_auc, calc_risk
from models.functions import return_predresults
from ue4nlp.ue_estimater_ensemble import UeEstimatorEnsemble
from ue4nlp.ue_estimater_trust import UeEstimatorTrustscore
from ue4nlp.ue_estimater_mcd import UeEstimatorDp


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/test1/configs", config_name="eval_config")
def main(cfg: DictConfig):
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Reg-torchlightning/pt{}/fold_{}/pred_results'.format(cfg.aes.prompt_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    
    results_dic = {}
    for ue_type in ['logvar', 'trust_score', 'mcdp_uncertainty', 'ense_uncertainty']: 
        five_fold_roc, five_fold_rcc, five_fold_rcc_y, five_fold_rpp = [], [], [], []
        for fold_result in five_fold_results:
            true = fold_result['labels']
            if ue_type == 'logvar':
                pred = fold_result['score']
                uncertainty = fold_result['calib_var']
            elif ue_type == 'mcdp_uncertainty':
                pred = fold_result['mcdp_score']
                uncertainty = fold_result['calib_mcdp_var']
            elif ue_type == 'ense_uncertainty':
                pred = fold_result['ense_score']
                uncertainty = fold_result['calib_ense_var']
            else:
                pred = fold_result['score']
                uncertainty = -fold_result['trust_score']

            risk = calc_risk(pred, true, cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
            rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
            rpp = calc_rpp(conf=-uncertainty, risk=risk)
            roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, prompt_id=cfg.aes.prompt_id)

            five_fold_rcc = np.append(five_fold_rcc, rcc_auc)
            five_fold_rpp = np.append(five_fold_rpp, rpp)
            five_fold_roc = np.append(five_fold_roc, roc_auc)
            five_fold_rcc_y.append(rcc_y)
        results_dic.update({ue_type: {'rcc': np.mean(five_fold_rcc).tolist(),
                                      'rpp': np.mean(five_fold_rcc).tolist(),
                                      'roc': np.mean(five_fold_roc).tolist(),
                                      'rcc_y': np.mean(np.array(five_fold_rcc_y), axis=0).tolist()}})
        
        
    with open(cfg.path.results_save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    

if __name__ == "__main__":
    main()