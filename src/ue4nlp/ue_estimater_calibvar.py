import numpy as np
import torch
from models.models import Scaler
from utils.cfunctions import regvarloss
from models.functions import return_predresults

class UeEstimatorCalibvar:
    def __init__(self, dev_labels: torch.Tensor, dev_score: torch.Tensor, dev_logvar: torch.Tensor):
        self.dev_labels = dev_labels
        self.dev_score = dev_score
        self.dev_logvar = dev_logvar
        
    def __call__(self, logvar: torch.Tensor):
        target_var = logvar.exp().sqrt().cuda()
        test_var = self.sigma_scaler(target_var).pow(2).to('cpu').detach().numpy().copy()
        return test_var
    
    def fit_ue(self):
        self.sigma_scaler = Scaler(init_S=1.0).cuda()
        s_opt = torch.optim.LBFGS([self.sigma_scaler.S], lr=3e-2, max_iter=2000)
        dev_std = self.dev_logvar.exp().sqrt().cuda()

        def closure():
            s_opt.zero_grad()
            loss = regvarloss(y_true=self.dev_labels.cuda(), y_pre_ave=self.dev_score.cuda(), y_pre_var=self.sigma_scaler(dev_std).pow(2).log().cuda())
            loss.backward()
            return loss
        s_opt.step(closure)