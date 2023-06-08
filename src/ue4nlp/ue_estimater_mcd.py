from models.functions import return_predresults
from ue4nlp.functions import compute_mulscore_mulvar, compute_mulprob_muluncertain
    

class UeEstimatorDp:
    def __init__(self, model, dropout_num, prompt_id, reg_or_class):
        self.model = model
        self.dropout_num = dropout_num
        self.prompt_id = prompt_id
        self.reg_or_class = reg_or_class
        
    def __call__(self, dataloader):
        mul_results = self._multi_pred(dataloader)
        mcdp_results = self._calc_var(mul_results)
        return mcdp_results
        
    def _multi_pred(self, dataloader):
        mul_results = {}
        for idx in range(self.dropout_num):
            pred_results = return_predresults(self.model, dataloader, rt_clsvec=False, dropout=True)
            del pred_results['labels']
            if len(mul_results) == 0:
                mul_results = {k: [v] for k, v in pred_results.items()}
            else:
                for (k1, v1), (k2, v2) in zip(mul_results.items(), pred_results.items()):
                  v1.append(v2)
        return mul_results
        
    def _calc_var(self, mul_results):
        mcdp_result = {}
        if self.reg_or_class == 'reg':
            mulscore, mulvar = compute_mulscore_mulvar(mul_results['score'], mul_results['logvar'], self.dropout_num)
            mcdp_result['mcdp_score'] = mulscore
            mcdp_result['mcdp_uncertainty'] = mulvar
        else:
            mulscore, muluncertainty = compute_mulprob_muluncertain(mul_results['logits'], self.dropout_num)
            mcdp_result['mcdp_score'] = mulscore
            mcdp_result['mcdp_uncertainty'] = muluncertainty
        return mcdp_result