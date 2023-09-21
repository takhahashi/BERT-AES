import os
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoTokenizer
from .utils.dataset import get_score_range, get_Dataset
from .utils.cfunctions import simple_collate_fn
from .models.models import Reg_class_mixmodel, Bert, EscoreScaler

    
@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/BERT-AES/configs", config_name="Escaler_train")
def main(cfg: DictConfig):
    para_savepath = cfg.path.savepath
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    train_dataset = get_Dataset('reg',
                                cfg.path.traindata_file_name,
                                cfg.aes.prompt_id,
                                tokenizer,
                                )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=4,
                                                    shuffle=True,
                                                    collate_fn=simple_collate_fn,
                                                    )
    low, high = get_score_range(cfg.aes.prompt_id)
    bert = Bert(cfg.model.model_name_or_path) 
    model = Reg_class_mixmodel(bert, high-low+1)
    model.load_state_dict(torch.load(cfg.path.model_save_path))

    model.eval()
    model = model.cuda()

    e_scaler = EscoreScaler(init_S=1.0).cuda()

    mseloss = nn.MSELoss()
    s_opt = torch.optim.LBFGS([e_scaler.S], lr=cfg.train.learning_rate, max_iter=cfg.train.max_iteration)
    all_class_pred = []
    all_reg_pred = []
    all_true_score = []

    train_loss = 0
    for idx, t_batch in enumerate(train_dataloader):
        batch = {k: v.cuda() for k, v in t_batch.items()}
        s_opt.zero_grad()

        int_score = torch.round(batch['labels'] * (high - low) + low).to(torch.float).cuda()
        outputs = {k: v.to('cpu').detach().numpy().copy() for k, v in model(batch).items()}
        class_pred_org = np.argmax(outputs['logits'],axis=1) + low
        reg_pred = outputs['score'].flatten()
        reg_pred_org = np.round(reg_pred * (high - low) + low)
        all_class_pred.append(class_pred_org)
        all_reg_pred.append(reg_pred_org)
        all_true_score.append(int_score)

    class_pred = torch.tensor(np.concatenate(all_class_pred)).cuda()
    reg_pred = torch.tensor(np.concatenate(all_reg_pred)).cuda()
    train_labels = torch.concat(all_true_score)


    def closure():
        s_opt.zero_grad()
        pred = e_scaler.left(class_pred) + e_scaler.right(reg_pred)
        loss = mseloss(pred, train_labels)
        loss.backward()
        return loss
    mean_pred = (class_pred + reg_pred) / 2
    noscale_loss = mseloss(mean_pred, train_labels)
    s_opt.step(closure)
    scale_pred = e_scaler.left(class_pred) + e_scaler.right(reg_pred)
    scale_loss = mseloss(scale_pred, train_labels)
    print(f'No_s:{noscale_loss}, Apply_S:{scale_loss}, S_Value:{torch.sigmoid(e_scaler.S)}')
    torch.save(e_scaler.state_dict(), para_savepath)

if __name__ == "__main__":
    main()