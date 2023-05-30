import torch.optim as optim

from models.models import (
    Bert,
    BertReg,
    BertClass
)

def create_module(reg_or_class, learning_rate, num_labels=None):
    bert = Bert()
    if reg_or_class == 'reg':
        model = BertReg(bert, learning_rate)
    elif reg_or_class == 'class':
        model = BertClass(bert, num_labels, learning_rate)
    else:
        raise ValueError("{} is not a valid value for reg_or_class".format(reg_or_class))
    return model    