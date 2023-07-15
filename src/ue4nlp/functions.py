import numpy as np
import torch
from utils.dataset import get_Dataset, get_score_range

def sep_features_by_class(X_features, scores):
    sep_features = {}
    for label in np.sort(np.unique(scores)):
        sep_features['{}'.format(label)] = X_features[scores == label]
    return sep_features

def diffclass_euclid_dist(target_feature, target_label, train_features_labels):
    min_dist = None
    for k, v in train_features_labels.items():
        if int(k) != target_label:
            for diff_vec in v:
                dist = np.linalg.norm(diff_vec-target_feature)
                if(min_dist is None or dist < min_dist):
                    min_dist = dist
    return min_dist

def sameclass_euclid_dist(target_feature, target_label, train_features_labels):
    min_dist = None
    for k, v in train_features_labels.items():
        if int(k) == target_label:
            for diff_vec in v:
                dist = np.linalg.norm(diff_vec-target_feature)
                if(min_dist is None or dist < min_dist):
                    min_dist = dist
    return min_dist

def compute_mulscore_mulvar(score, logvar, mulnum):
    vars = [np.exp(i) for i in logvar]
    sumvar = np.sum(vars, axis=0)
    sumscore = np.sum(score, axis=0)
    powerscore = np.power(score, 2)
    sumpower = np.sum(powerscore, axis=0)

    mulscore = np.divide(sumscore, mulnum)
    mulvar = np.divide(sumvar + sumpower, mulnum) - np.power(mulscore, 2)
    return mulscore, mulvar

def compute_mulprob_epiuncertain(numpy_logits, mulnum):
    soft_fn = torch.nn.Softmax(dim=2)
    logits = torch.tensor(numpy_logits)
    pred_probs = soft_fn(logits)
    assert pred_probs.shape[0] == mulnum

    mean_probs = torch.mean(pred_probs, dim=0)
    mean_entro = -torch.sum(torch.log2(mean_probs) * mean_probs, dim=-1)
    all_entro = -torch.sum(torch.sum(torch.log2(pred_probs) * pred_probs, dim=0), dim=-1)

    torch_scores = torch.argmax(mean_probs, dim=-1)
    torch_uncertainty = mean_entro - all_entro / mulnum

    return torch_scores.numpy(), torch_uncertainty.numpy()

def compute_mulMP(numpy_logits, mulnum):
    soft_fn = torch.nn.Softmax(dim=2)
    logits = torch.tensor(numpy_logits)
    pred_probs = soft_fn(logits)
    assert pred_probs.shape[0] == mulnum

    mean_probs = torch.mean(pred_probs, dim=0)
    mulMP = mean_probs[torch.arange(len(mean_probs)), torch.argmax(mean_probs, dim=-1)]
    torch_scores = torch.argmax(mean_probs, dim=-1)

    return torch_scores.numpy(), mulMP.numpy()


def compute_mulEntropy(numpy_logits, mulnum):
    soft_fn = torch.nn.Softmax(dim=2)
    logits = torch.tensor(numpy_logits)
    pred_probs = soft_fn(logits)
    assert pred_probs.shape[0] == mulnum

    mean_probs = torch.mean(pred_probs, dim=0)
    mean_entro = -torch.sum(torch.log2(mean_probs) * mean_probs, dim=-1)
    torch_scores = torch.argmax(mean_probs, dim=-1)
    
    return torch_scores.numpy(), mean_entro.numpy()

def compute_MixMulMP(score, numpy_logits, mulnum, prompt_id):
    low, high = get_score_range(prompt_id)
    sumscore = np.sum(score, axis=0)
    mulscore = np.divide(sumscore, mulnum)

    soft_fn = torch.nn.Softmax(dim=2)
    logits = torch.tensor(numpy_logits)
    pred_probs = soft_fn(logits)
    mean_probs = torch.mean(pred_probs, dim=0)

    pred_int_score = torch.tensor(np.round(mulscore * (high - low)), dtype=torch.int32)
    MixMulMP = mean_probs[torch.arange(len(mean_probs)), pred_int_score]
    return mulscore, MixMulMP