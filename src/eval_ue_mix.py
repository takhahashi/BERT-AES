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
from utils.ue_metric_func import calc_rcc_auc, calc_rpp, calc_roc_auc, calc_risk, calc_rcc_auc_scaledrmse, calc_rcc_auc_rmse
from models.functions import return_predresults
from ue4nlp.ue_estimater_ensemble import UeEstimatorEnsemble
from ue4nlp.ue_estimater_trust import UeEstimatorTrustscore
from ue4nlp.ue_estimater_mcd import UeEstimatorDp

def calc_mean_rcc_y(rcc_y_lis):
    min_len = len(rcc_y_lis[0])
    for rcc_y in rcc_y_lis:
        if len(rcc_y) < min_len:
            min_len = len(rcc_y)
    rcc_y_arr = []
    for rcc_y in rcc_y_lis:
        rcc_y_arr.append(np.array(rcc_y)[:min_len])
    mean_rcc_y = np.mean(rcc_y_arr, axis=0)
    return mean_rcc_y.tolist()

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/BERT-AES/configs", config_name="eval_ue_config")
def main(cfg: DictConfig):

    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Mix-torchlightning/prompt{}/fold_{}/pred_results_org_loss_50'.format(cfg.aes.prompt_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    save_dir_path = cfg.path.save_dir_path

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix conf####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['score']
        uncertainty = -foldr['mix_conf']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(pred, true, -uncertainty, cfg.rcc.metric_type, cfg.aes.prompt_id, reg_or_class='reg', num_el=25, binary_risk=True)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    mean_rcc_y = calc_mean_rcc_y(fresults_rcc_y)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': mean_rcc_y}
    save_path = save_dir_path + '/mix_org_loss_50'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Mix-torchlightning/prompt{}/fold_{}/pred_results_org_loss_100'.format(cfg.aes.prompt_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    save_dir_path = cfg.path.save_dir_path

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix conf####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['score']
        uncertainty = -foldr['mix_conf']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(pred, true, -uncertainty, cfg.rcc.metric_type, cfg.aes.prompt_id, reg_or_class='reg', num_el=25, binary_risk=True)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    mean_rcc_y = calc_mean_rcc_y(fresults_rcc_y)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': mean_rcc_y}
    save_path = save_dir_path + '/mix_org_loss_100'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Mix-torchlightning/prompt{}/fold_{}/pred_results_org_loss_1000'.format(cfg.aes.prompt_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    save_dir_path = cfg.path.save_dir_path

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix conf####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['score']
        uncertainty = -foldr['mix_conf']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(pred, true, -uncertainty, cfg.rcc.metric_type, cfg.aes.prompt_id, reg_or_class='reg', num_el=25, binary_risk=True)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    mean_rcc_y = calc_mean_rcc_y(fresults_rcc_y)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': mean_rcc_y}
    save_path = save_dir_path + '/mix_org_loss_1000'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

if __name__ == "__main__":
    main()