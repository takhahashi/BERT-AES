import torch.nn as nn
from cfunctions import regvarloss

def load_criterion(criterion_type):
    if criterion_type == 'regvarloss':
        return regvarloss
    elif criterion_type == 'mseloss':
        return nn.MSELoss()
    elif criterion_type == 'crossentropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("{} is not a valid value for criterion_type".format(criterion_type))