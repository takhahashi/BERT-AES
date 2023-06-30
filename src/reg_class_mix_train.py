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
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    train_dataset = get_Dataset('reg',
                                cfg.path.traindata_file_name,
                                cfg.aes.prompt_id,
                                tokenizer,
                                )
    dev_dataset = get_Dataset('reg',
                            cfg.path.valdata_file_name,
                            cfg.aes.prompt_id,
                            tokenizer,
                            )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=cfg.training.batch_size,
                                                    shuffle=True,
                                                    collate_fn=simple_collate_fn,
                                                    )
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,
                                                    batch_size=cfg.training.batch_size,
                                                    shuffle=True,
                                                    collate_fn=simple_collate_fn,
                                                    )
    
    low, high = get_score_range(cfg.aes.prompt_id)
    model = Bertratermean(high-low+1)
    model.train()
    model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    num_train_batch = len(train_dataloader)
    num_dev_batch = len(dev_dataloader)
    earlystopping = EarlyStopping(patience=3, path = cfg.path.save_path, verbose = True)

    model.train()
    crossentropy = nn.CrossEntropyLoss()
    mseloss = nn.MSELoos()

    trainloss_list, devloss_list = [], []
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(0, 15):
        lossall = 0
        devlossall = 0
        model.train()
        for data in train_dataloader:
            data = {k: v.cuda() for k, v in data.items()}
            int_score = torch.round(data['labels'] * (high - low) + low).to(torch.int32).type(torch.LongTensor)
            with torch.cuda.amp.autocast():
                outputs = model(data)
                crossentropy_el = crossentropy(outputs['logits'], int_score)
                mseloss_el = mseloss(outputs['score'], data['labels'])
                loss = crossentropy_el + mseloss_el
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            lossall += loss.to('cpu').detach().numpy().copy()

        trainloss_list = np.append(trainloss_list, lossall/num_train_batch)
        # dev QWKの計算
        
        model.eval()
        for dev_data in dev_dataloader:
            d_data = {k: v.cuda() for k, v in dev_data.items()}
            int_score = torch.round(d_data['labels'] * (high - low) + low).to(torch.int32).type(torch.LongTensor)
            dev_outputs = {k: v.to('cpu').detach() for k, v in model(d_data).items()}
            crossentropy_el = crossentropy(dev_outputs['logits'], int_score)
            mseloss_el = mseloss(dev_outputs['score'], d_data['labels'])
            devlossall += crossentropy_el + mseloss_el
        devloss_list = np.append(devloss_list, devlossall/num_dev_batch)

        print(f'Epoch:{epoch}, train_Loss:{lossall/num_train_batch:.4f}, dev_loss:{devlossall/num_dev_batch:.4f}')
        earlystopping(devlossall/num_dev_batch, model)
        if(earlystopping.early_stop == True): break


if __name__ == "__main__":
    main()