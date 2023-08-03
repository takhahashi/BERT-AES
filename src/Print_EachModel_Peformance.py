from utils.acc_metric_func import calc_qwk
from utils.cfunctions import simple_collate_fn, score_f2int
from utils.dataset import get_Dataset, get_score_range
from utils.cfunctions import simple_collate_fn
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from models.models import Reg_class_mixmodel, Bert
from models.functions import return_predresults
from utils.ue_metric_func import calc_rcc_auc, calc_rpp, calc_roc_auc, calc_risk
import numpy as np

prompt_id=7
low, high = get_score_range(prompt_id)
for fold in [1, 2, 3]:
  test_dataset = get_Dataset('reg',
                            '/content/drive/MyDrive/GoogleColab/1.AES/ASAP/data/fold_{}/test.tsv'.format(fold),
                            prompt_id,
                            AutoTokenizer.from_pretrained('bert-base-uncased'),
                            )
  test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=16,
                                            shuffle=False,
                                            collate_fn=simple_collate_fn,
                                            )
  for id in range(5):
    model_save_path = '/content/drive/MyDrive/GoogleColab/1.AES/ASAP/Mix-torchlightning/prompt{}/fold_{}/id{}'.format(prompt_id, fold, id)
    bert = Bert('bert-base-uncased')
    model = Reg_class_mixmodel(bert, high-low+1)
    model.load_state_dict(torch.load(model_save_path))
    eval_results = return_predresults(model, test_dataloader, rt_clsvec=False, dropout=False)
    softmax = nn.Softmax(dim=1)
    pred_int_score = torch.tensor(np.round(eval_results['score'] * (high - low)), dtype=torch.int32)
    pred_probs = softmax(torch.tensor(eval_results['logits']))
    mix_trust = pred_probs[torch.arange(len(pred_probs)), pred_int_score]
    eval_results.update({'mix_conf': mix_trust.numpy().copy()})

    true = score_f2int(eval_results['labels'], prompt_id)
    pred = score_f2int(eval_results['score'], prompt_id)
    uncertainty = -eval_results['mix_conf']
    risk = calc_risk(pred, true, 'reg', prompt_id, binary=True)
    rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
    rpp = calc_rpp(conf=-uncertainty, risk=risk)
    roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', prompt_id=prompt_id)
    print(f'fold:{fold}, id:{id}, corr:{np.corrcoef(true, pred)[0][1]}, qwk:{calc_qwk(true, pred, prompt_id, "reg")}, roc:{roc_auc}, rpp:{rpp}, rcc:{rcc_auc}')