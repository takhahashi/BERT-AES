import os
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from utils.utils_data import TrainDataModule
from utils.dataset import get_score_range, get_Dataset, get_asap2_dataset
from utils.cfunctions import simple_collate_fn, theta_collate_fn, simplevar_ratersd_loss
from utils.utils_models import create_module
from models.functions import return_predresults
from models.models import Bertratermean
from utils.cfunctions import regvarloss, EarlyStopping
from models.models import Scaler

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP 2/BERT-AES/configs", config_name="asap2_reg_config")
def main(cfg: DictConfig):
    batch_num = cfg.training.batch_size
    savepath = cfg.path.save_path

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = get_asap2_dataset(pd.read_table(cfg.path.train_file_name, sep='\t'), scoretype='ratermean_ratersd', tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_num, shuffle=True, drop_last=False, collate_fn=theta_collate_fn)

    dev_dataset = get_asap2_dataset(pd.read_table(cfg.path.dev_file_name, sep='\t'), scoretype='ratermean_ratersd', tokenizer=tokenizer)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_num, shuffle=False, drop_last=False, collate_fn=theta_collate_fn)

    model = Bertratermean()
    model.train()
    model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    train_batchnum = len(train_dataloader)
    dev_batchnum = len(dev_dataloader)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    earlystopping = EarlyStopping(patience=3, path = savepath, verbose = True)

    trainloss_list, devloss_list = [], []
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(0, 15):
        lossall = 0
        devlossall = 0
        model.train()
        for data in train_dataloader:
            data = {k: v.cuda() for k, v in data.items()}
            rater_logvar = data['sd'].pow(2).log()
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = simplevar_ratersd_loss(data['score'], rater_logvar, outputs['score'], outputs['logvar'])
            lossall += loss.to('cpu').detach().numpy().copy()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

        trainloss_list = np.append(trainloss_list, lossall/train_batchnum)
        # dev QWKの計算
        
        model.eval()
        for dev_data in dev_dataloader:
            d_data = {k: v.cuda() for k, v in dev_data.items()}
            rater_logvar = d_data['sd'].pow(2).log().to('cpu').detach()
            dev_outputs = {k: v.to('cpu').detach() for k, v in model(d_data).items()}
            dev_loss = simplevar_ratersd_loss(d_data['score'].to('cpu').detach(), rater_logvar, dev_outputs['score'], dev_outputs['logvar'])
            devlossall += dev_loss
        devloss_list = np.append(devloss_list, devlossall/dev_batchnum)

        print(f'Epoch:{epoch}, train_Loss:{lossall/train_batchnum:.4f}, dev_loss:{devlossall/dev_batchnum:.4f}')
        earlystopping(devlossall/dev_batchnum, model)
        if(earlystopping.early_stop == True): break


if __name__ == "__main__":
    main()