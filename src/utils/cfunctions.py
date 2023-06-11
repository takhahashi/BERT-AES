import torch
import numpy as np
from utils.dataset import get_score_range
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score, roc_auc_score

def regvarloss(y_true, y_pre_ave, y_pre_var):
    loss = torch.exp(-torch.flatten(y_pre_var))*torch.pow(y_true - torch.flatten(y_pre_ave), 2)/2 + torch.flatten(y_pre_var)/2
    loss = torch.sum(loss)
    return loss

def simple_collate_fn(list_of_data):
  pad_max_len = torch.tensor(0)
  for data in list_of_data:
    if(torch.count_nonzero(data['attention_mask']) > pad_max_len):
      pad_max_len = torch.count_nonzero(data['attention_mask'])
  in_ids, token_type, atten_mask, labels = [], [], [], []
  for data in list_of_data:
    in_ids.append(data['input_ids'][:pad_max_len])
    token_type.append(data['token_type_ids'][:pad_max_len])
    atten_mask.append(data['attention_mask'][:pad_max_len])
    labels.append(data['labels'])
  batched_tensor = {}
  batched_tensor['input_ids'] = torch.stack(in_ids)
  batched_tensor['token_type_ids'] = torch.stack(token_type)
  batched_tensor['attention_mask'] = torch.stack(atten_mask)
  batched_tensor['labels'] = torch.tensor(labels)
  return batched_tensor

def score_f2int(score, prompt_id):
  low, high = get_score_range(prompt_id)
  return np.round(score * (high - low) + low).astype('int32')