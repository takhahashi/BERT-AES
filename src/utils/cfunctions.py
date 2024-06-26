import torch
import numpy as np
from utils.dataset import get_score_range
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score, roc_auc_score
from collections import defaultdict

class DynamicWeightAverage:
  def __init__(self, num_tasks, temp):
    self.num_tasks = num_tasks
    self.temp = temp
    self.loss_log = []
    for _ in range(self.num_tasks):
       self.loss_log.append([])

  def __call__(self, *args):
    weights = self._calc_weights()
    all_loss = 0
    for idx, loss in enumerate(args):
      self.loss_log[idx].append(loss.to('cpu').detach().numpy().copy())
    for w, l in zip(weights, args):
       all_loss += w * l
    return all_loss, weights.to('cpu').tolist()
  
  def _calc_weights(self):
    if len(self.loss_log[0]) < 2:
      w_lis = np.ones(self.num_tasks)
    else:
      w_lis = np.array([l[-1]/l[-2] for l in self.loss_log])
    exp_w = np.exp(w_lis/self.temp)
    return torch.tensor(self.num_tasks*exp_w/np.sum(exp_w)).cuda()

class ScaleDiffBalance:
  def __init__(self, task_names, priority=None, beta=1.):
    self.task_names = task_names
    self.num_tasks = len(self.task_nemes)
    self.task_priority = {}
    if priority is not None:
        for k, v in priority.items():
           self.task_priority[k] = v
    else:
        for k in self.task_names:
          self.task_priority[k] =  1/self.num_tasks
    self.all_loss_log = []
    self.loss_log = defaultdict(list)
    self.beta = beta
  
  def update(self, all_loss, *args, **kwargs):
    self.all_loss_log = np.append(self.all_loss_log, all_loss)
    for k, v in kwargs.items():
       self.loss_log[k] = np.append(self.loss_log[k], v)
  
  def __call__(self, *args, **kwargs):
    scale_weights = self._calc_scale_weights()
    diff_weights = self._calc_diff_weights()
    alpha = self._calc_alpha(diff_weights)
    all_loss = 0
    for k, each_loss in kwargs:
       all_loss += scale_weights[k] * diff_weights[k] * each_loss
    if len(self.all_loss_log) < 1:
      pre_loss = 0
    else:
      pre_loss = self.all_loss_log[-1]
    return alpha * all_loss, scale_weights, diff_weights, alpha, pre_loss
  
  def _calc_scale_weights(self):
    w_dic = {}
    if len(self.all_loss_log) < 1:
      for k, v in self.task_priority.items():
         w_dic[k] = v.cuda()
    else:
      for k, each_task_loss_arr in self.loss_log.items():
         task_priority = self.task_priority[k]
         w_dic[k] = torch.tensor(self.all_loss_log[-1]*task_priority/each_task_loss_arr[-1]).cuda()
    return w_dic
  
  def _calc_diff_weights(self):
    w_dic = {}
    if len(self.all_loss_log) < 2:
      for k, _ in self.task_priority.items():
         w_dic[k] = torch.tensor(1.).cuda()
    else:
      for k, each_task_loss_arr in self.loss_log.items():
         w_dic[k] = torch.tensor(((each_task_loss_arr[-1]/each_task_loss_arr[-2])/(self.all_loss_log[-1]/self.all_loss_log[-2]))**self.beta).cuda()
    return w_dic
  
  def _calc_alpha(self, diff_weights):
    if len(self.all_loss_log) < 2:
      return torch.tensor(1.).cuda()
    else:
      tmp = 0
      for k, _ in self.task_priority.items():
         tmp += self.task_priority[k].cuda() * diff_weights[k]
      return (1/tmp).cuda()
      

def regvarloss(y_true, y_pre_ave, y_pre_var):
    loss = torch.exp(-torch.flatten(y_pre_var))*torch.pow(y_true - torch.flatten(y_pre_ave), 2)/2 + torch.flatten(y_pre_var)/2
    loss = torch.sum(loss)
    return loss

def simplevar_ratersd_loss(y_true, rater_var, y_pre_ave, y_pre_var):
    loss = torch.exp(-torch.flatten(y_pre_var))*torch.pow(y_true - torch.flatten(y_pre_ave), 2)/2 + torch.flatten(y_pre_var)/2 + (rater_var - torch.flatten(y_pre_var)) ** 2
    loss = torch.sum(loss)
    return loss

def mix_loss(y_trues, y_preds, logits, high, low, alpha):
   mse_loss, cross_loss = 0, 0
   y_trues_org = np.round(torch.flatten(y_trues).to('cpu').detach().numpy().copy() * (high - low))
   probs = logits.softmax(dim=1)[list(range(len(y_trues_org))), y_trues_org]
   loss = alpha*((torch.flatten(y_trues) - torch.flatten(y_preds)) ** 2)*probs - probs
   mse_loss = torch.sum((torch.flatten(y_trues) - torch.flatten(y_preds)) ** 2)
   cross_loss = torch.sum(-torch.log(probs))
   loss = torch.sum(loss)
   return loss, mse_loss, cross_loss

def mix_loss1(y_trues, y_preds, logits, high, low, alpha): #  \frac{\|\hat{y}-y\|^2}{-\log{\hat{P}_{y}}}-\log{\hat{P}_{y}}
   mse_loss, cross_loss = 0, 0
   y_trues_org = np.round(torch.flatten(y_trues).to('cpu').detach().numpy().copy() * (high - low))
   probs = logits.softmax(dim=1)[list(range(len(y_trues_org))), y_trues_org]
   neg_ln_probs = -torch.log(probs)
   loss = alpha * (((torch.flatten(y_trues) - torch.flatten(y_preds)) ** 2)/neg_ln_probs) + neg_ln_probs
   mse_loss = torch.sum((torch.flatten(y_trues) - torch.flatten(y_preds)) ** 2)
   cross_loss = torch.sum(-torch.log(probs))
   loss = torch.sum(loss)
   return loss, mse_loss, cross_loss

def mix_loss2(y_trues, y_preds, logits, high, low, alpha): #  -\hat{P}_{y} + \hat{P}_{y}\|\hat{y}-y\|^2
   mse_loss, cross_loss = 0, 0
   y_trues_org = np.round(torch.flatten(y_trues).to('cpu').detach().numpy().copy() * (high - low))
   probs = logits.softmax(dim=1)[list(range(len(y_trues_org))), y_trues_org]
   loss = -probs + probs * ((torch.flatten(y_trues) - torch.flatten(y_preds)) ** 2)
   mse_loss = torch.sum((torch.flatten(y_trues) - torch.flatten(y_preds)) ** 2)
   cross_loss = torch.sum(-torch.log(probs))
   loss = torch.sum(loss)
   return loss, mse_loss, cross_loss

def mix_loss3(y_trues, y_preds, logits, high, low): #  -\hat{P}_{y} + \hat{P}_{y}\|\hat{y}-y\|^2
   mse_loss, cross_loss = 0, 0
   y_trues_org = np.round(torch.flatten(y_trues).to('cpu').detach().numpy().copy() * (high - low))
   y_preds_org = np.round(torch.flatten(y_preds).to('cpu').detach().numpy().copy() * (high - low))
   correct_probs = logits.softmax(dim=1)[y_trues_org == y_preds_org, y_trues_org[y_trues_org == y_preds_org]]
   wrong_probs = logits.softmax(dim=1)[y_trues_org != y_preds_org]

   correct_ln_probs = -torch.log(correct_probs)
   wrong_ln_probs = torch.mean(-torch.log(wrong_probs), dim=-1)
   ln_probs = torch.concat([correct_ln_probs, wrong_ln_probs])

   mse_loss = torch.mean((torch.flatten(y_trues) - torch.flatten(y_preds)) ** 2)
   cross_loss = torch.mean(ln_probs)
   normal_cross_loss = torch.mean(correct_ln_probs)

   #print(f'y_true:{y_trues_org}, y_pred:{y_preds_org}, probs:{logits.softmax(dim=1)}')
   #print(f'correct_probs:{correct_probs}, wrong_probs:{wrong_probs}')
   #print(f'corr_ln_probs:{correct_ln_probs}, wro_ln_probs:{wrong_ln_probs}, cat_ln_probs:{ln_probs}')
   return mse_loss, cross_loss, normal_cross_loss

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
  if labels[0].shape  != torch.tensor(1).shape:
     batched_tensor['labels'] = torch.stack(labels)
  else:
     batched_tensor['labels'] = torch.tensor(labels)
  return batched_tensor

def theta_collate_fn(list_of_data):
  pad_max_len = torch.tensor(0)
  for data in list_of_data:
    if(torch.count_nonzero(data['attention_mask']) > pad_max_len):
      pad_max_len = torch.count_nonzero(data['attention_mask'])
  in_ids, token_type, atten_mask, score, sd = [], [], [], [], []
  for data in list_of_data:
    in_ids.append(data['input_ids'][:pad_max_len])
    token_type.append(data['token_type_ids'][:pad_max_len])
    atten_mask.append(data['attention_mask'][:pad_max_len])
    score.append(data['score'])
    sd.append(data['sd'])
  batched_tensor = {}
  batched_tensor['input_ids'] = torch.stack(in_ids)
  batched_tensor['token_type_ids'] = torch.stack(token_type)
  batched_tensor['attention_mask'] = torch.stack(atten_mask)
  batched_tensor['score'] = torch.tensor(score)
  batched_tensor['sd'] = torch.tensor(sd)
  return batched_tensor

def ratermean_collate_fn(list_of_data):
  pad_max_len = torch.tensor(0)
  for data in list_of_data:
    if(torch.count_nonzero(data['attention_mask']) > pad_max_len):
      pad_max_len = torch.count_nonzero(data['attention_mask'])
  in_ids, token_type, atten_mask, score = [], [], [], []
  for data in list_of_data:
    in_ids.append(data['input_ids'][:pad_max_len])
    token_type.append(data['token_type_ids'][:pad_max_len])
    atten_mask.append(data['attention_mask'][:pad_max_len])
    score.append(data['score'])
  batched_tensor = {}
  batched_tensor['input_ids'] = torch.stack(in_ids)
  batched_tensor['token_type_ids'] = torch.stack(token_type)
  batched_tensor['attention_mask'] = torch.stack(atten_mask)
  batched_tensor['score'] = torch.tensor(score)
  return batched_tensor

def score_f2int(score, prompt_id):
  low, high = get_score_range(prompt_id)
  return np.round(score * (high - low) + low).astype('int32')

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
            print(f'first_score: {-self.best_score}.     Saving model ...')
            torch.save(model.state_dict(), self.path)
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