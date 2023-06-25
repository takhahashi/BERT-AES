import numpy as np
from sklearn.metrics import roc_auc_score

from utils.cfunctions import score_f2int

def calc_rcc_auc(conf, risk):
  n = len(conf)
  cr_pair = list(zip(conf, risk))
  cr_pair.sort(key=lambda x: x[0], reverse=True)
  cumulative_risk = [cr_pair[0][1]]
  for i in range(1, n):
    cumulative_risk.append(cr_pair[i][1] + cumulative_risk[-1])
  points_x, points_y = [], []
  auc = 0
  for k in range(n):
    auc += cumulative_risk[k] / (1+k)
    points_x.append((1+k) / n)  # coverage
    points_y.append(cumulative_risk[k] / (1+k))  # current avg. risk
  return auc, points_x, points_y

def calc_rpp(conf, risk):
  n = len(conf)
  cr_pair = list(zip(conf, risk))
  cr_pair.sort(key=lambda x: x[0], reverse=False)

  cnt = 0
  for i in range(n):
    for j in range(i, n):
      if(cr_pair[i][1] < cr_pair[j][1]):
        cnt += 1
  return cnt / (n**2)

def calc_roc_auc(pred, true, conf, reg_or_class, prompt_id):
  if reg_or_class == 'reg':
    int_scores = score_f2int(pred, prompt_id)
    int_true = score_f2int(true, prompt_id)
  else:
    int_scores = pred
    int_true = true
  return roc_auc_score(int_scores == int_true, conf)

def calc_risk(pred, true, reg_or_class, prompt_id, binary=False):
  if binary == True:
    if reg_or_class =='reg':
      int_scores = score_f2int(pred, prompt_id)
      int_true = score_f2int(true, prompt_id)
    else:
      int_scores = pred.astype('int32')
      int_true = true.astype('int32')
    return int_scores != int_true
  else:
    return (pred - true) ** 2
    