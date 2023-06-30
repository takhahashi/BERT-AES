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
from utils.cfunctions import simple_collate_fn, score_f2int
from utils.utils_models import create_module
from utils.dataset import get_Dataset
from utils.ue_metric_func import calc_rcc_auc, calc_rpp, calc_roc_auc, calc_risk
from utils.acc_metric_func import calc_qwk
from models.functions import return_predresults
from ue4nlp.ue_estimater_ensemble import UeEstimatorEnsemble
from ue4nlp.ue_estimater_trust import UeEstimatorTrustscore
from ue4nlp.ue_estimater_mcd import UeEstimatorDp
from sklearn.metrics import cohen_kappa_score


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/BERT-AES/configs", config_name="eval_acc_config")
def main(cfg: DictConfig):
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Reg-torchlightning/pt{}/fold_{}/pred_results'.format(cfg.aes.prompt_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    save_dir_path = cfg.path.save_dir_path
    prompt_id = cfg.aes.prompt_id


    corr_arr, qwk_arr, rmse_arr = [], [], []
    ##simple reg####
    for foldr in five_fold_results:
        true = score_f2int(foldr['labels'], prompt_id)
        pred = score_f2int(foldr['score'], prompt_id)

        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, calc_qwk(true, pred, prompt_id, 'reg'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/simple_reg_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    


    corr_arr, qwk_arr, rmse_arr = [], [], []
    ##dp reg####
    for foldr in five_fold_results:
        true = score_f2int(foldr['labels'], prompt_id)
        pred = score_f2int(foldr['mcdp_score'], prompt_id)

        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, calc_qwk(true, pred, prompt_id, 'reg'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/dp_reg_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)



    corr_arr, qwk_arr, rmse_arr = [], [], []
    ##ense reg####
    for foldr in five_fold_results:
        true = score_f2int(foldr['labels'], prompt_id)
        pred = score_f2int(foldr['ense_score'], prompt_id)

        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, calc_qwk(true, pred, prompt_id, 'reg'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/ense_reg_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    ##class###
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/Y15/{}_results/Class_{}/fold{}'.format(cfg.sas.question_id, cfg.sas.score_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})


    corr_arr, qwk_arr, rmse_arr = [], [], []
    ##simple_class####
    for foldr in five_fold_results:
        true = foldr['labels'].astype('int32')
        pred = np.argmax(foldr['logits'], axis=-1).astype('int32')

        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, calc_qwk(true, pred, prompt_id, 'class'), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/simple_class_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    corr_arr, qwk_arr, rmse_arr = [], [], []
    ##dp_class####
    for foldr in five_fold_results:
        true = foldr['labels'].astype('int32')
        pred = foldr['mcdp_score'].astype('int32')

        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, calc_qwk(true, pred, prompt_id, 'class'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/dp_class_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    corr_arr, qwk_arr, rmse_arr = [], [], []
    ##ense_class####
    for foldr in five_fold_results:
        true = foldr['labels'].astype('int32')
        pred = foldr['ense_score'].astype('int32')

        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, calc_qwk(true, pred, prompt_id, 'class'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/ense_class_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

if __name__ == "__main__":
    main()