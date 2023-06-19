import numpy as np
import torch
from models.models import Scaler
from utils.cfunctions import regvarloss
from models.functions import return_predresults

class UeEstimatorCalibvar:
    def __init__(self, model, dev_dataloader):
        self.model = model
        self.dev_dataloader = dev_dataloader
        
    def __call__(self, dataloader):
        test_results = return_predresults(self.model, dataloader, rt_clsvec=False, dropout=False)
        test_var = self.sigma_scaler(torch.tensor(test_results['logvar']).exp().sqrt().cuda()).pow(2).to('cpu').detach().numpy().copy()
        return {'calib_var': test_var}
    
    def fit_ue(self):
        self.sigma_scaler = Scaler(init_S=1.0).cuda()
        s_opt = torch.optim.LBFGS([self.sigma_scaler.S], lr=3e-2, max_iter=2000)
        self.model.eval()
        with torch.no_grad():
            dev_results = return_predresults(self.model, self.dev_dataloader, rt_clsvec=False, dropout=False)
        dev_mu = torch.tensor(dev_results['score']).cuda()
        dev_std = torch.tensor(dev_results['logvar']).exp().sqrt().cuda()
        dev_labels = torch.tensor(dev_results['labels']).cuda()

        def closure():
            s_opt.zero_grad()
            loss = regvarloss(y_true=dev_labels, y_pre_ave=dev_mu, y_pre_var=self.sigma_scaler(dev_std).pow(2).log())
            loss.backward()
            return loss
        s_opt.step(closure)