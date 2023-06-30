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


def get_theta(scores_array):
	norm_scores = (scores_array + 3) / 6
	norm_scores[norm_scores > 1.0] = 1.
	norm_scores[norm_scores < 0] = 0.
	return norm_scores

def get_ratermean(scores_array):
  scores_mean = np.mean(scores_array, where = scores_array!=-1, axis=-1)
  return np.round(scores_mean) / 4.

def get_classratermean(scores_array):
  scores_mean = np.round(np.mean(scores_array, where = scores_array!=-1, axis=-1))
  return scores_mean.astype('int32')

class ThetaDataSet:
  def __init__(self, X, Y):
    self.input_ids = X['input_ids']
    self.token_type_ids = X['token_type_ids']
    self.attention_mask = X['attention_mask']
    self.score = Y['score']
    self.sd = Y['sd']

  def __len__(self):
    return len(self.score)

  def __getitem__(self, index):
    return {'input_ids': self.input_ids[index], 
		    'token_type_ids': self.token_type_ids[index],
            'attention_mask': self.attention_mask[index], 
            'score': self.score[index],
			'sd': self.sd[index]}

class RaterDataSet:
  def __init__(self, X, Y):
    self.input_ids = X['input_ids']
    self.token_type_ids = X['token_type_ids']
    self.attention_mask = X['attention_mask']
    self.score = Y

  def __len__(self):
    return len(self.score)

  def __getitem__(self, index):
    return {'input_ids': self.input_ids[index], 
		    'token_type_ids': self.token_type_ids[index],
            'attention_mask': self.attention_mask[index], 
            'score': self.score[index]}
 
def get_asap2_dataset(dataf, scoretype, tokenizer):
  x = dataf["essay"].tolist()
  encoding = tokenizer(x, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
  if scoretype == 'irttheta':
    theta = get_theta(np.array(dataf["theta"])).tolist()
    sd = np.array(dataf["sd"]).tolist()
    labels = {'theta':torch.tensor(theta, dtype=torch.float32), 'sd':torch.tensor(sd, dtype=torch.float32)}
    return ThetaDataSet(encoding, labels)
  elif scoretype == 'ratermean_irtsd':
    ratermean = get_ratermean(np.array(dataf.iloc[:, 4:42])).tolist()
    sd = np.array(dataf["sd"]).tolist()
    labels = {'score':torch.tensor(ratermean, dtype=torch.float32), 'sd':torch.tensor(sd, dtype=torch.float32)}
    return ThetaDataSet(encoding, labels)
  elif scoretype == 'ratermean_ratersd':
    ratermean = get_ratermean(np.array(dataf.iloc[:, 4:42])).tolist()
    sd = np.array([np.var(scorearray[scorearray != -1]) for scorearray in np.array(dataf.iloc[:, 4:42])])
    labels = {'score':torch.tensor(ratermean, dtype=torch.float32), 'sd':torch.tensor(np.sqrt(sd), dtype=torch.float32)}
    return ThetaDataSet(encoding, labels)
  elif scoretype == 'regratermean':
    score = get_ratermean(np.array(dataf.iloc[:, 4:42])).tolist()
    return RaterDataSet(encoding, score)
  elif scoretype == 'classratermean':
    score = get_classratermean(np.array(dataf.iloc[:, 4:42])).tolist()
    return RaterDataSet(encoding, score)
  else:
    raise ValueError("{} is not a valid value for scoretype".format(scoretype))