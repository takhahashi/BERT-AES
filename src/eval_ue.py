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
    ##reg###
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Reg-torchlightning/pt{}/fold_{}/pred_results'.format(cfg.aes.prompt_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    save_dir_path = cfg.path.save_dir_path
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##simple var####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['score']
        uncertainty = foldr['calib_var']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        #rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
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
    save_path = save_dir_path + '/simplevar'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    """
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], [] 
    ##reg_trust_score####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['score']
        uncertainty = -foldr['trust_score']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/reg_trust_score'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##reg_dp###
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score']
        uncertainty = foldr['calib_mcdp_var']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/reg_dp'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    """
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##reg_ense###
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = foldr['ense_var']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(pred, true, -uncertainty, cfg.rcc.metric_type, cfg.aes.prompt_id, reg_or_class='reg')
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
    save_path = save_dir_path + '/reg_mul'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class###
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Class-torchlightning/pt{}/fold_{}/pred_results'.format(cfg.aes.prompt_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    save_dir_path = cfg.path.save_dir_path

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##MP####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = np.argmax(foldr['logits'], axis=-1)
        uncertainty = -foldr['MP']
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(pred, true, -uncertainty, cfg.rcc.metric_type, cfg.aes.prompt_id, reg_or_class='class', num_el=25, binary_risk=True)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    mean_rcc_y = calc_mean_rcc_y(fresults_rcc_y)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': mean_rcc_y}
    save_path = save_dir_path + '/MP'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class trust####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = np.argmax(foldr['logits'], axis=-1)
        uncertainty = -foldr['trust_score']
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(pred, true, -uncertainty, cfg.rcc.metric_type, cfg.aes.prompt_id, reg_or_class='class', num_el=25, binary_risk=True)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    mean_rcc_y = calc_mean_rcc_y(fresults_rcc_y)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': mean_rcc_y}
    save_path = save_dir_path + '/class_trust_score'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    """
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class dp MP####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score'] 
        uncertainty = -foldr['mcdp_MP']
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/class_dp_MP'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class dp Ent###
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score'] 
        uncertainty = foldr['mcdp_entropy'] 
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/class_dp_entropy'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class dp Epistemic####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score'] 
        uncertainty = foldr['mcdp_epi_uncertainty']
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/class_dp_epistemic'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    """
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class mul MP####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = -foldr['ense_MP']
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(pred, true, -uncertainty, cfg.rcc.metric_type, cfg.aes.prompt_id, reg_or_class='class')
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    mean_rcc_y = calc_mean_rcc_y(fresults_rcc_y)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': mean_rcc_y}
    save_path = save_dir_path + '/class_mul_MP'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    """
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class mul Ent####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = foldr['ense_entropy']
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/class_mul_entropy'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class mul####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = foldr['ense_epi_uncertainty']
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/class_mul_epistemic'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    """

    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/GP-git/pt{}/fold_{}/pred_results'.format(cfg.aes.prompt_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##GP####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['score']
        uncertainty = foldr['std']
        risk = calc_risk(pred, true, 'gp', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        #rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(pred, true, -uncertainty, cfg.rcc.metric_type, cfg.aes.prompt_id, reg_or_class='gp', num_el=25, binary_risk=True)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='gp', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    mean_rcc_y = calc_mean_rcc_y(fresults_rcc_y)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': mean_rcc_y}
    save_path = save_dir_path + '/GP'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/GP-git/pt{}/fold_{}/pred_results_spectralnorm'.format(cfg.aes.prompt_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##GP_spectralnorm####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['score']
        uncertainty = foldr['std']
        risk = calc_risk(pred, true, 'gp', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        #rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(pred, true, -uncertainty, cfg.rcc.metric_type, cfg.aes.prompt_id, reg_or_class='gp', num_el=25, binary_risk=True)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='gp', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    mean_rcc_y = calc_mean_rcc_y(fresults_rcc_y)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': mean_rcc_y}
    save_path = save_dir_path + '/GP_spectralnorm'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    """
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Mix-torchlightning/prompt{}/fold_{}/pred_results'.format(cfg.aes.prompt_id, fold)) as f:
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
    save_path = save_dir_path + '/mix'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix_dp###
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score']
        uncertainty = -foldr['mcdp_MP']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/mix_dp'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix_dp_ent##
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score']
        uncertainty = foldr['mcdp_entropy']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/mix_dp_entropy'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##MIx_Ense##
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = -foldr['ense_MP']
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
    save_path = save_dir_path + '/mix_mul'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix_Ense_entropy
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = foldr['ense_entropy']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/mix_mul_entropy'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    """
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Mix-torchlightning/prompt{}/fold_{}/pred_results_scale_only'.format(cfg.aes.prompt_id, fold)) as f:
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
    save_path = save_dir_path + '/mix_scale_only'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    """
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix_dp###
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score']
        uncertainty = -foldr['mcdp_MP']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/mix_dp_scale_only'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix_dp_ent##
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score']
        uncertainty = foldr['mcdp_entropy']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/mix_dp_entropy_scale_only'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    """
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##MIx_Ense##
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = -foldr['ense_MP']
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
    save_path = save_dir_path + '/mix_mul_scale_only'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    """
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix_Ense_entropy
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = foldr['ense_entropy']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/mix_mul_entropy_scale_only'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix MP####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['score']
        uncertainty = -foldr['MP']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=True)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/mixMP'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    """
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Mix-torchlightning/prompt{}/fold_{}/pred_results_expected_score_expected_score'.format(cfg.aes.prompt_id, fold)) as f:
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
    save_path = save_dir_path + '/mix_expected_score'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    """
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix_dp###
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score']
        uncertainty = -foldr['mcdp_MP']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/mix_dp_scale_only'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix_dp_ent##
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score']
        uncertainty = foldr['mcdp_entropy']
        risk = calc_risk(pred, true, 'reg', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/mix_dp_entropy_scale_only'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    """
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##MIx_Ense##
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = -foldr['ense_MP']
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
    save_path = save_dir_path + '/mix_mul_expected_score'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Mix-torchlightning/prompt{}/fold_{}/pred_results_weighted_exp_score'.format(cfg.aes.prompt_id, fold)) as f:
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
    save_path = save_dir_path + '/mix_weighted_exp_score'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)



    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##ordinal_reg###
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Ord_reg-torchlightning/pt{}/fold_{}/pred_results'.format(cfg.aes.prompt_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    save_dir_path = cfg.path.save_dir_path

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##ordinal_reg####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = np.argmax(foldr['logits'], axis=-1)
        uncertainty = -foldr['MP']
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(pred, true, -uncertainty, cfg.rcc.metric_type, cfg.aes.prompt_id, reg_or_class='class', num_el=25, binary_risk=True)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    mean_rcc_y = calc_mean_rcc_y(fresults_rcc_y)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': mean_rcc_y}
    save_path = save_dir_path + '/ordinal_reg'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    """
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##ordinal_reg trust####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = np.argmax(foldr['logits'], axis=-1)
        uncertainty = -foldr['trust_score']
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(pred, true, -uncertainty, cfg.rcc.metric_type, cfg.aes.prompt_id, reg_or_class='class', num_el=25, binary_risk=True)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    mean_rcc_y = calc_mean_rcc_y(fresults_rcc_y)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': mean_rcc_y}
    save_path = save_dir_path + '/ordinal_reg_trust_score'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class dp MP####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score'] 
        uncertainty = -foldr['mcdp_MP']
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/class_dp_MP'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class dp Ent###
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score'] 
        uncertainty = foldr['mcdp_entropy'] 
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/class_dp_entropy'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class dp Epistemic####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score'] 
        uncertainty = foldr['mcdp_epi_uncertainty']
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/class_dp_epistemic'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    """
    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##ordinal_reg mul MP####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = -foldr['ense_MP']
        risk = calc_risk(pred, true, 'class', cfg.aes.prompt_id, binary=cfg.ue.binary_risk)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(pred, true, -uncertainty, cfg.rcc.metric_type, cfg.aes.prompt_id, reg_or_class='class')
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', prompt_id=cfg.aes.prompt_id)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    mean_rcc_y = calc_mean_rcc_y(fresults_rcc_y)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': mean_rcc_y}
    save_path = save_dir_path + '/ordinal_reg_mul_MP'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Mix-torchlightning/prompt{}/fold_{}/pred_results_org_loss'.format(cfg.aes.prompt_id, fold)) as f:
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
    save_path = save_dir_path + '/mix_org_loss'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##MIx_Ense##
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = -foldr['ense_MP']
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
    save_path = save_dir_path + '/ense_mix_org_loss'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
if __name__ == "__main__":
    main()