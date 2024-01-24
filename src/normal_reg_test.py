import os

import json
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoTokenizer
from utils.cfunctions import simple_collate_fn
from utils.utils_models import create_module
from utils.dataset import get_Dataset, get_score_range
from models.functions import return_predresults
from transformers import AutoModel

class Bert_reg(nn.Module):
    def __init__(self, model_name_or_path, lr):
        super(Bert_reg, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        self.lr = lr

        nn.init.normal_(self.linear.weight, std=0.02)  # 重みの初期化

    def forward(self, dataset):
        outputs = self.bert(dataset['input_ids'], token_type_ids=dataset['token_type_ids'], attention_mask=dataset['attention_mask'])
        last_hidden_state = outputs['last_hidden_state'][:, 0, :]
        score = self.sigmoid(self.linear(last_hidden_state))
        return {'score': score}



@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/BERT-AES/configs", config_name="normal_reg_train")
def main(cfg: DictConfig):
    test_dataset = get_Dataset(cfg.model.reg_or_class,
                                cfg.path.testdata_file_name,
                                cfg.aes.prompt_id,
                                AutoTokenizer.from_pretrained(cfg.model.model_name_or_path),
                                )
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=cfg.training.batch_size,
                                                shuffle=False,
                                                collate_fn=simple_collate_fn,
                                                )

    model = Bert_reg(
        cfg.model.model_name_or_path,
        cfg.training.learning_rate,
        )
    model.load_state_dict(torch.load(cfg.path.model_save_path))

    eval_results = return_predresults(model, test_dataloader, rt_clsvec=False, dropout=False)
    list_results = {k: v.tolist() for k, v in eval_results.items() if type(v) == type(np.array([1, 2, 3.]))}
    
    with open(cfg.path.results_save_path, mode="wt", encoding="utf-8") as f:
        json.dump(list_results, f, ensure_ascii=False)
    

if __name__ == "__main__":
    main()