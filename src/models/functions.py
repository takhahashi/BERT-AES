import numpy as np
import torch

def return_predresults(model, test_dataloader, rt_clsvec, dropout = False):
    model = model.cuda()
    if dropout == False:
        model.eval()
        if rt_clsvec == True:
            bert = model.bert
            bert.eval()
            bert = bert.cuda()
    else:
        model.train()
        if rt_clsvec == True:
            bert = model.bert
            bert.train()
            bert = bert.cuda()

    eval_results = {}
    for t_data in test_dataloader:
        batch = {k: v.cuda() for k, v in t_data.items()}
        y_true = {'labels': batch['labels'].to('cpu').detach().numpy().copy()}
        x = {'input_ids':batch['input_ids'],
                    'attention_mask':batch['attention_mask'],
                    'token_type_ids':batch['token_type_ids']}
        outputs = {k: v.to('cpu').detach().numpy().copy() for k, v in model(x).items()}
        if rt_clsvec == True:
            cls_outputs = {k: v.to('cpu').detach().numpy().copy() for k, v in bert(x).items()}

        if len(eval_results) == 0:
            eval_results.update(y_true)
            eval_results.update(outputs)
            if rt_clsvec == True:
                eval_results.update(cls_outputs)
        else:
            y_true.update(outputs)
            if rt_clsvec == True:
                y_true.update(cls_outputs)
            eval_results = {k1: np.concatenate([v1, v2]) for (k1, v1), (k2, v2) in zip(eval_results.items(), y_true.items())}

    try:
        eval_results['score'] = eval_results['score'].flatten()
        eval_results['logvar'] = eval_results['logvar'].flatten()
    except:
        print()
    return eval_results

def extract_clsvec_truelabels(model, dataloader):
    bert = model.bert
    bert.eval()
    bert = bert.cuda()
    eval_results = {}
    for t_data in dataloader:
        batch = {k: v.cuda() for k, v in t_data.items()}
        y_true = {'labels': batch['labels'].to('cpu').detach().numpy().copy()}
        x = {'input_ids':batch['input_ids'],
                'attention_mask':batch['attention_mask'],
                'token_type_ids':batch['token_type_ids']}

        cls_outputs = {k: v.to('cpu').detach().numpy().copy() for k, v in bert(x).items()}

        if len(eval_results) == 0:
            eval_results.update(cls_outputs)
            eval_results.update(y_true)
        else:
            cls_outputs.update(y_true)
            eval_results = {k1: np.concatenate([v1, v2]) for (k1, v1), (k2, v2) in zip(eval_results.items(), cls_outputs.items())}
    return eval_results['hidden_state'], eval_results['labels']


def extract_clsvec_predlabels(model, dataloader):
    model.eval()
    model = model.cuda()
    bert = model.bert
    bert.eval()
    bert = bert.cuda()
    eval_results = {}
    for t_data in dataloader:
        batch = {k: v.cuda() for k, v in t_data.items()}
        y_true = {'labels': batch['labels'].to('cpu').detach().numpy().copy()}
        x = {'input_ids':batch['input_ids'],
                    'attention_mask':batch['attention_mask'],
                    'token_type_ids':batch['token_type_ids']}
        outputs = {k: v.to('cpu').detach().numpy().copy() for k, v in model(x).items()}
        cls_outputs = {k: v.to('cpu').detach().numpy().copy() for k, v in bert(x).items()}

        if len(eval_results) == 0:
            eval_results.update(y_true)
            eval_results.update(outputs)
            eval_results.update(cls_outputs)
        else:
            y_true.update(outputs)
            y_true.update(cls_outputs)
            eval_results = {k1: np.concatenate([v1, v2]) for (k1, v1), (k2, v2) in zip(eval_results.items(), y_true.items())}

    for k, v in eval_results.items():
        if k == 'score':
            score = v.flatten()
            return eval_results['hidden_state'], score
        else:
            score = np.argmax(v, axis=-1)
            return eval_results['hidden_state'], score