import os
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
from utils.dataset import get_score_range, get_Dataset
from utils.cfunctions import simple_collate_fn
from utils.utils_models import create_module
from models.functions import return_predresults
from utils.cfunctions import regvarloss
from models.models import Scaler

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path
    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分

        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.val_loss_min = -score
            print(f'first_score: {-self.best_score}')
            #self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score <= self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/BERT-AES/configs", config_name="reg_config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    train_dataset = get_Dataset(cfg.model.reg_or_class,
                                cfg.path.traindata_file_name,
                                cfg.aes.prompt_id,
                                tokenizer,
                                )
    dev_dataset = get_Dataset(cfg.model.reg_or_class,
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

    model = create_module(
        cfg.model.model_name_or_path,
        cfg.model.reg_or_class,
        cfg.training.learning_rate,
        )
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    model = model.cuda()
    earlystopping = EarlyStopping(patience=cfg.training.patience, verbose=True, path=cfg.path.save_path)

    scaler = torch.cuda.amp.GradScaler()
    sigma_scaler = Scaler(init_S=1.0).cuda()

    num_train_batch = len(train_dataloader)
    num_dev_batch = len(dev_dataloader)
    for epoch in range(cfg.training.n_epochs):
        train_loss_all = dev_loss_all = 0
        for idx, t_batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in t_batch.items()}
            with torch.cuda.amp.autocast():
                training_step_outputs = model.training_step(batch, idx)
            scaler.scale(training_step_outputs['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

            train_loss_all = training_step_outputs['loss'].to('cpu').detach().numpy().copy()


        ###calibrate_step###calibrate_step###
        model.eval()
        with torch.no_grad():
            dev_results = return_predresults(model, dev_dataloader, rt_clsvec=False, dropout=False)
        dev_mu = torch.tensor(dev_results['score']).cuda()
        dev_std = torch.tensor(dev_results['logvar']).exp().sqrt().cuda()
        dev_labels = torch.tensor(dev_results['labels']).cuda()

        # find optimal S
        s_opt = torch.optim.LBFGS([sigma_scaler.S], lr=3e-2, max_iter=2000)

        def closure():
            s_opt.zero_grad()
            loss = regvarloss(y_true=dev_labels, y_pre_ave=dev_mu, y_pre_var=sigma_scaler(dev_std).pow(2).log())
            loss.backward()
            return loss
        s_opt.step(closure)

        for idx, d_batch in enumerate(dev_dataloader):
            batch = {k: v.cuda() for k, v in d_batch.items()}
            dev_step_outputs = model.validation_step(batch, idx)
            dev_mu = dev_step_outputs['score']
            dev_std = dev_step_outputs['logvar'].exp().sqrt()
            dev_labels = dev_step_outputs['labels']
            dev_loss_all += regvarloss(y_true=dev_labels, y_pre_ave=dev_mu, y_pre_var=sigma_scaler(dev_std.cuda()).pow(2).log()).to('cpu').detach().numpy().copy()

        print(f'Epoch:{epoch}, train_loss:{train_loss_all/num_train_batch}, dev_loss:{dev_loss_all/num_dev_batch}')
        earlystopping(dev_loss_all, model)
        if earlystopping.early_stop == True:
            break


if __name__ == "__main__":
    main()