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
from models.models import Reg_class_mixmodel, Bert
from utils.cfunctions import regvarloss, EarlyStopping, DynamicWeightAverage, ScaleDiffBalance, mix_loss1, mix_loss3
from models.models import Scaler
import matplotlib.pyplot as plt
import wandb

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/BERT-AES/configs", config_name="reg_class_mix")
def main(cfg: DictConfig):

    wandb.init(project=cfg.wandb.project, 
               name=cfg.wandb.project_name,)
               #settings=wandb.Settings(start_method="thread"))

    
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
    bert = Bert(cfg.model.model_name_or_path)
    model = Reg_class_mixmodel(bert, high-low+1)
    model.train()
    model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    num_train_batch = len(train_dataloader)
    num_dev_batch = len(dev_dataloader)
    earlystopping = EarlyStopping(patience=cfg.training.patience, path = cfg.path.save_path, verbose = True)

    model.train()
    crossentropy = nn.CrossEntropyLoss()
    mseloss = nn.MSELoss()
    weight_d = ScaleDiffBalance(num_tasks=2, beta=1.)

    trainloss_list, devloss_list, dev_mse_list, dev_cross_list = [], [], [], []
    scaler = torch.cuda.amp.GradScaler()
    """
    lossall, mse_lossall = 0, 0
    for data in train_dataloader:
        data = {k: v.cuda() for k, v in data.items()}
        outputs = model(data)
        loss, mse_loss, cross_loss = mix_loss1(data['labels'].squeeze(), outputs['score'].squeeze(), outputs['logits'], high, low, alpha=1.)

        lossall += loss.to('cpu').detach().numpy().copy()
        mse_lossall += mse_loss.to('cpu').detach().numpy().copy()
    """
    mse_weights = cfg.training.original_loss_weight

    for epoch in range(cfg.training.n_epochs):
        model.train()
        mse_loss_list, cross_loss_list = [], []
        lossall, cross_lossall, mse_lossall, normal_cross_lossall = 0, 0, 0, 0
        s_weiall, diff_weiall = 0, 0
        devlossall = 0
        for data in train_dataloader:
            data = {k: v.cuda() for k, v in data.items()}
            int_score = torch.round(data['labels'] * (high - low)).to(torch.int32).type(torch.LongTensor).cuda()
            with torch.cuda.amp.autocast():
                outputs = model(data)
                """
                crossentropy_el = crossentropy(outputs['logits'], int_score)
                mseloss_el = mseloss(outputs['score'].squeeze(), data['labels'])
                loss, s_wei, diff_wei, alpha, pre_loss = weight_d(crossentropy_el, mseloss_el)
                """
                mse_loss, cross_loss, normal_cross_loss = mix_loss3(data['labels'].squeeze(), outputs['score'].squeeze(), outputs['logits'], high, low)
                loss, s_wei, diff_wei, alpha, pre_loss = weight_d(mse_loss, cross_loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            lossall += loss.to('cpu').detach().numpy().copy()
            mse_lossall += mse_loss.to('cpu').detach().numpy().copy()
            cross_lossall += cross_loss.to('cpu').detach().numpy().copy()
            normal_cross_lossall += normal_cross_loss.to('cpu').detach().numpy().copy()
        # dev QWKの計算
        
        model.eval()
        for dev_data in dev_dataloader:
            d_data = {k: v.cuda() for k, v in dev_data.items()}
            int_score = torch.round(d_data['labels'] * (high - low)).to(torch.int32).type(torch.LongTensor).to('cpu')
            dev_outputs = {k: v.to('cpu').detach() for k, v in model(d_data).items()}
            #crossentropy_el = crossentropy(dev_outputs['logits'], int_score)
            #mseloss_el = mseloss(dev_outputs['score'].squeeze(), d_data['labels'].to('cpu').detach())
            #loss, s_wei, diff_wei, alpha, pre_loss = weight_d(crossentropy_el, mseloss_el)
            mse_loss, cross_loss, normal_cross_loss = mix_loss3(data['labels'].squeeze(), outputs['score'].squeeze(), outputs['logits'], high, low)
            loss, s_wei, diff_wei, alpha, pre_loss = weight_d(mse_loss, cross_loss)
            devlossall += loss.to('cpu').detach().numpy().copy()
        s_wei = weight_d._calc_scale_weights()
        diff_wei = weight_d._calc_diff_weights()
        wandb.log({"epoch":epoch+0.001,
                   "all_loss":lossall/num_train_batch, 
                   "dev_loss":devlossall/num_dev_batch,
                   "mse_weighted":diff_wei[0] * s_wei[0] * mse_lossall/num_train_batch,
                   "cross_weited":diff_wei[1] * s_wei[1] * cross_lossall/num_train_batch,
                   })
        devloss_list = np.append(devloss_list, devlossall/num_dev_batch)
        #dev_mse_list = np.append(dev_mse_list, mseloss_el)
        #dev_cross_list = np.append(dev_cross_list, crossentropy_el)
        

        """
        wandb.log({
            "all_loss":lossall/num_train_batch,
            "Scale_Weight_mse":s_wei[1], 
            "Scale_Weight_cross":s_wei[0],
            "mse_loss":mse_loss/num_train_batch, 
            "cross_loss":cross_loss/num_train_batch,
            "Scaled_mse_loss":s_wei[1] * mse_loss/num_train_batch,
            "Scaled_cross_loss":s_wei[0] * cross_loss/num_train_batch,
            "Diff_Weight_mse":diff_wei[1],
            "Diff_Weight_cross":diff_wei[0],
        })
        """
        weight_d.update(lossall/num_train_batch, mse_lossall/num_train_batch, cross_lossall/num_train_batch)
        print(f'Epoch:{epoch}, train_Loss:{lossall/num_train_batch:.4f}, dev_loss:{devlossall/num_dev_batch:.4f}')
        earlystopping(devlossall/num_dev_batch, model)
        if(earlystopping.early_stop == True): break
    wandb.finish()
    """
    # Plot trainloss_list in blue
    plt.plot(trainloss_list, color='blue', label='Train Loss')

    # Plot devloss_list in red
    plt.plot(devloss_list, color='red', label='Dev Loss')

    # Set labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Development Loss')

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Plot loss1_list in the first subplot
    ax1.plot(dev_mse_list, color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mse_loss')

    # Plot loss2_list in the second subplot
    ax2.plot(dev_cross_list, color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('cross_entro')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    """

if __name__ == "__main__":
    main()