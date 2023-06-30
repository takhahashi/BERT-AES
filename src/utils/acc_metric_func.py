from sklearn.metrics import cohen_kappa_score
from utils.dataset import get_score_range

def calc_qwk(true, pred, prompt_id, reg_or_class):
    low, high = get_score_range(prompt_id)
    if reg_or_class == 'reg':
        return cohen_kappa_score(true, pred, labels = list(range(low, high + 1)), weights='quadratic')
    elif reg_or_class == 'class':
        return cohen_kappa_score(true, pred, labels = list(range(high - low + 1)), weights='quadratic')