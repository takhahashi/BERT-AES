import os
import hydra
import torch
import gpytorch
import numpy as np
from omegaconf import DictConfig
from transformers import AutoTokenizer
from utils.dataset import get_Dataset, get_score_range
from utils.cfunctions import simple_collate_fn
from utils.utils_models import create_module
from models.functions import extract_clsvec_truelabels
from models.models import GPModel
import json

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/BERT-AES/configs", config_name="GP_eval")
def main(cfg: DictConfig):
    train_dataset = get_Dataset('class', 
                                cfg.path.train_data_file_name, 
                                cfg.aes.prompt_id, 
                                AutoTokenizer.from_pretrained(cfg.scoring_model.model_name_or_path),
                                )
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=8,
                                                  shuffle=False,
                                                  collate_fn=simple_collate_fn,
                                                  )
    
    test_dataset = get_Dataset('class', 
                                cfg.path.test_data_file_name, 
                                cfg.aes.prompt_id, 
                                AutoTokenizer.from_pretrained(cfg.scoring_model.model_name_or_path),
                                )
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=8,
                                                  shuffle=False,
                                                  collate_fn=simple_collate_fn,
                                                  )
    
    low, high = get_score_range(cfg.aes.prompt_id)
    num_labels = high - low + 1
    if cfg.scoring_model.spectral_norm == True:
       #scoring_model_path = cfg.path.scoring_model_savepath + '_sepctralnorm'
       results_save_path = cfg.path.results_save_path + '_spectralnorm'
    else:
       #scoring_model_path = cfg.path.scoring_model_savepath
       results_save_path = cfg.path.results_save_path
    classifier = create_module(cfg.scoring_model.model_name_or_path,
                                cfg.scoring_model.reg_or_class,
                                learning_rate=1e-5,
                                num_labels=num_labels,
                                spectral_norm=cfg.scoring_model.spectral_norm,
                                )
    classifier = classifier.cuda()
    classifier.load_state_dict(torch.load(cfg.path.scoring_model_savepath), strict=False)
    classifier.eval()

    word_vec, labels = extract_clsvec_truelabels(classifier, train_dataloader)
    train_x = torch.FloatTensor(word_vec)
    train_y = torch.FloatTensor(labels)

    word_vec, labels = extract_clsvec_truelabels(classifier, test_dataloader)
    test_x = torch.FloatTensor(word_vec)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    model.load_state_dict(torch.load(cfg.path.GPmodel_save_path))
    likelihood.eval()
    model.eval()    

    predictions = model(test_x)
    mean = predictions.mean.cpu().detach().numpy()
    std = predictions.stddev.cpu().detach().numpy()

    eval_results = {'labels':labels, 'score':mean, 'std':std}
    list_results = {k: v.tolist() for k, v in eval_results.items() if type(v) == type(np.array([1, 2, 3.]))}
    
    with open(results_save_path, mode="wt", encoding="utf-8") as f:
        json.dump(list_results, f, ensure_ascii=False)

if __name__ == "__main__":
    main()