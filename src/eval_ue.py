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
            if ue_type == 'mcdp_uncertainty':
                pred = fold_result['mcdp_score']
            elif ue_type == 'ense_uncertainty':
                pred = fold_result['ense_score']
            else:
                pred = fold_result['score']
            pred = fold_result['score']
            true = fold_result['labels']

            if ue_type == 'trust_score':
                uncertainty = -fold_results[ue_type]
            else:
                uncertainty = fold_result[ue_type]
                
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
        



    
    five
    
    five_fold_results
    cfg.fold.prompt_id = prompt_id
    cfg.fold
    cfg.fold.prompt_id=prompt_id
    cfg.aes.prompt_id
    roc_auc
    rcc_auc
    rpp


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