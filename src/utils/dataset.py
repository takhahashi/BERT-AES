import numpy as np
import pandas as pd
import torch

asap_ranges = {
	0: (0, 60),
	1: (2,12),
	2: (1,6),
	3: (0,3),
	4: (0,3),
	5: (0,4),
	6: (0,4),
	7: (0,30),
	8: (0,60)
}

max_length = 512

def get_score_range(prompt_id):
	return asap_ranges[prompt_id]

def get_model_friendly_scores(scores_array, prompt_id_array):
	arg_type = type(prompt_id_array)
	if arg_type is int:
		low, high = asap_ranges[prompt_id_array]
		scores_array = (scores_array - low) / (high - low)
	else:
		dim = scores_array.shape[0]
		low = np.zeros(dim)
		high = np.zeros(dim)
		for ii in range(dim):
			low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
		scores_array = (scores_array - low) / (high - low)
	return scores_array

class DataSet:
  def __init__(self, X, Y):
    self.input_ids = X['input_ids']
    self.attention_mask = X['attention_mask']
    self.token_type_ids = X['token_type_ids']
    self.labels = Y

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return {'input_ids': self.input_ids[index], 
            'attention_mask': self.attention_mask[index], 
            'token_type_ids': self.token_type_ids[index], 
            'labels': self.labels[index]}

def get_Dataset(reg_or_class, datapath, prompt_id, tokenizer):
	low, high = asap_ranges[prompt_id]
	dataf = pd.read_table(datapath, sep='\t')
	dat_p = dataf[dataf["essay_set"] == prompt_id]
	x = dat_p["essay"].tolist()
	encoding = tokenizer(x, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
	if reg_or_class == 'reg':
		y = get_model_friendly_scores(np.array(dat_p["domain1_score"]), prompt_id).tolist()
		labels = torch.tensor(y, dtype=torch.float32)
		return DataSet(encoding, labels)
	elif reg_or_class == 'class':
		y = (np.array(dat_p["domain1_score"]) - low).tolist()
		labels = torch.tensor(y, dtype=torch.int32)
		return DataSet(encoding, labels.type(torch.LongTensor))
	else:
		raise ValueError("{} is not a valid value for reg_or_class".format(reg_or_class))