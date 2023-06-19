from models.functions import return_predresults
from ue4nlp.functions import compute_mulscore_mulvar, compute_mulprob_muluncertain
from utils.utils_models import create_module
from ue4nlp.ue_estimater_calibvar import UeEstimatorCalibvar

import torch
    

class UeEstimatorEnsemble:
    def __init__(self, model_paths, prompt_id, reg_or_class):
        self.model_paths = model_paths
        self.prompt_id = prompt_id
        self.reg_or_class = reg_or_class
        
    def __call__(self, dataloader):
        ense_results = self._predict_with_multimodel(dataloader)
        logvar = torch.tensor(ense_results['ense_var']).log()
        calib_var_results = self._calib_var(logvar)
        return ense_results.update({'calib_ense_var': calib_var_results})
    
    def _multi_pred(self, dataloader):
        mul_results = {}
        for model_path in self.model_paths:
            model = create_module(
                'bert-base-uncased',
                self.reg_or_class,
                0.00005,
                save_path = model_path,
                )
            pred_result = return_predresults(model, dataloader, rt_clsvec=False, dropout=False)
            del pred_result['labels']
            if len(mul_results) == 0:
                mul_results = {k: [v] for k, v in pred_result.items()}
            else:
                for (k1, v1), (k2, v2) in zip(mul_results.items(), pred_result.items()):
                  v1.append(v2)
        return mul_results


    def _predict_with_multimodel(self, dataloader):
        mul_pred_results = self._multi_pred(dataloader)
        mul_num = len(self.model_paths)
        ense_result = {}
        if self.reg_or_class == 'reg':
            mulscore, mulvar = compute_mulscore_mulvar(mul_pred_results['score'], mul_pred_results['logvar'], mul_num)
            ense_result['ense_score'] = mulscore
            ense_result['ense_var'] = mulvar
        else:
            mulscore, muluncertainty = compute_mulprob_muluncertain(mul_pred_results['logits'], mul_num)
            ense_result['ense_score'] = mulscore
            ense_result['ense_var'] = muluncertainty
        return ense_result
    
    def _calib_var(self, logvar: torch.Tensor):
        calib_var_estimater = UeEstimatorCalibvar(self.model,
                                                self.dev_dataloader,
                                                )
        calib_var_estimater.fit_ue()
        calib_var_results = calib_var_estimater(logvar)
        return calib_var_results
