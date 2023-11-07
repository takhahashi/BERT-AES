import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from utils.dataset import get_score_range

from utils.cfunctions import score_f2int

def calc_rcc_auc(true, pred, conf, metric_type, prompt_id, reg_or_class=None, num_el=None, binary_risk=None):
  if metric_type == 'normal':
    risk = calc_risk(pred, true, 'reg', prompt_id, binary=binary_risk)
    auc, points_x, points_y = calc_rcc_auc_simple(conf, risk)
  elif metric_type == 'rmse':
    auc, points_x, points_y = calc_rcc_auc_rmse(pred, true, conf, prompt_id, reg_or_class)
  elif metric_type == 'scaledrmse':
    auc, points_x, points_y = calc_rcc_auc_scaledrmse(pred, true, conf, prompt_id, reg_or_class)
  elif metric_type == 'qwk':
    auc, points_x, points_y = calc_rcc_auc_qwk(pred, true, conf, prompt_id, reg_or_class, num_el)
  else:
    raise ValueError("{} is not a valid value for metric_type".format(metric_type))
  return auc, points_x, points_y

def calc_rcc_auc_simple(conf, risk):
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

def calc_rcc_auc_rmse(pred, true, conf, prompt_id, reg_or_class):
  low, high = get_score_range(prompt_id)
  if reg_or_class == 'reg':
    pred_org = score_f2int(pred, prompt_id)
    true_org = score_f2int(true, prompt_id)
  else:
    pred_org = (pred + low).astype('int32')
    true_org = (true + low).astype('int32')
  risk = (pred_org - true_org) ** 2
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

def calc_rcc_auc_scaledrmse(pred, true, conf, prompt_id, reg_or_class):
  low, high = get_score_range(prompt_id)
  if reg_or_class == 'reg':
    pred_org = score_f2int(pred, prompt_id)
    true_org = score_f2int(true, prompt_id)
  elif reg_or_class == 'class':
    pred_org = (pred + low).astype('int32')
    true_org = (true + low).astype('int32')
  elif reg_or_class == 'gp':
    pred_org = np.round(pred + low).astype('int32')
    true_org = np.round(true + low).astype('int32')
  print(pred_org)
  print(true_org)
  print('------------')
  pred_scaled = (pred_org - low) / (high - low)
  true_scaled = (true_org - low) / (high - low)
  risk = (pred_scaled - true_scaled) ** 2
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

def calc_rcc_auc_qwk(pred, true, conf, prompt_id, reg_or_class, num_el):
  score_fn = cohen_kappa_score
  n = len(conf)
  low, high = get_score_range(prompt_id)
  if reg_or_class == 'reg':
    pred_sorted = score_f2int(pred[np.argsort(-conf)], prompt_id)
    true_sorted = score_f2int(true[np.argsort(-conf)], prompt_id)
  else:
    pred_sorted = (pred[np.argsort(-conf)] + low).astype('int32')
    true_sorted = (true[np.argsort(-conf)] + low).astype('int32')
  conf_sorted = conf[np.argsort(-conf)]
  ptc_pair = list(zip(pred_sorted, true_sorted, conf_sorted))
  points_x, points_y = [0], [0]
  auc = 0
  for k in range(0, n, num_el):
    if k+num_el <= n:
      pred_subset = [i[0] for i in ptc_pair[0:k+num_el]]
      true_subset = [i[1] for i in ptc_pair[0:k+num_el]]
      score_fn_subset = 1 - cohen_kappa_score(true_subset, pred_subset, labels = list(range(low, high + 1)), weights='quadratic')
      if np.isnan(score_fn_subset):
        score_fn_subset = 0.
      auc += (points_y[-1] + score_fn_subset) * num_el /2
      points_x.append(n/(k+num_el))
      points_y.append(score_fn_subset)
    else:
      pred_subset = [i[0] for i in ptc_pair[k:]]
      true_subset = [i[1] for i in ptc_pair[k:]]
      score_fn_subset = 1 - cohen_kappa_score(true_sorted, pred_sorted, labels = list(range(low, high + 1)), weights='quadratic')
      auc += (points_y[-1] + score_fn_subset) * (n%num_el) /2
      points_x.append(1.)
      points_y.append(score_fn_subset)
  return auc, points_x, points_y

def calc_rcc_auc_corr(pred, true, conf, prompt_id, reg_or_class, num_el):
  n = len(conf)
  low, high = get_score_range(prompt_id)
  if reg_or_class == 'reg':
    pred_sorted = score_f2int(pred[np.argsort(-conf)], prompt_id)
    true_sorted = score_f2int(true[np.argsort(-conf)], prompt_id)
  else:
    pred_sorted = (pred[np.argsort(-conf)] + low).astype('int32')
    true_sorted = (true[np.argsort(-conf)] + low).astype('int32')
  conf_sorted = conf[np.argsort(-conf)]
  ptc_pair = list(zip(pred_sorted, true_sorted, conf_sorted))
  points_x, points_y = [0], [0]
  auc = 0
  for k in range(0, n, num_el):
    if k+num_el <= n:
      pred_subset = [i[0] for i in ptc_pair[0:k+num_el]]
      true_subset = [i[1] for i in ptc_pair[0:k+num_el]]
      score_fn_subset = 1 - np.corrcoef(true_subset, pred_subset)[0][1]
      if np.isnan(score_fn_subset):
        score_fn_subset = 0.
      auc += (points_y[-1] + score_fn_subset) * num_el /2
      points_x.append(n/(k+num_el))
      points_y.append(score_fn_subset)
    else:
      pred_subset = [i[0] for i in ptc_pair[k:]]
      true_subset = [i[1] for i in ptc_pair[k:]]
      score_fn_subset = 1 - np.corrcoef(true_sorted, pred_sorted)[0][1]
      auc += (points_y[-1] + score_fn_subset) * (n%num_el) /2
      points_x.append(1.)
      points_y.append(score_fn_subset)
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
  elif reg_or_class == 'class':
    int_scores = pred
    int_true = true
  elif reg_or_class == 'gp':
    int_scores = np.round(pred).astype('int32')
    int_true = true.astype('int32')
  return roc_auc_score(int_scores == int_true, conf)

def calc_risk(pred, true, reg_or_class, prompt_id, binary=False):
  if binary == True:
    if reg_or_class =='reg':
      int_scores = score_f2int(pred, prompt_id)
      int_true = score_f2int(true, prompt_id)
    elif reg_or_class == 'class':
      int_scores = pred.astype('int32')
      int_true = true.astype('int32')
    elif reg_or_class == 'gp':
      int_scores = np.round(pred).astype('int32')
      int_true = np.round(true).astype('int32')
    return (int_scores != int_true).astype('int32')
  else:
    if reg_or_class =='reg':
      int_scores = score_f2int(pred, prompt_id)
      int_true = score_f2int(true, prompt_id)
    else:
      int_scores = pred.astype('int32')
      int_true = true.astype('int32')
    return (int_scores-int_true)**2
    