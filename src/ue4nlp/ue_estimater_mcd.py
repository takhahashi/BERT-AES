from models.functions import return_predresults
from ue4nlp.functions import compute_mulscore_mulvar, compute_mulMP, compute_mulEntropy, compute_mulprob_epiuncertain, compute_MixMulMP
from ue4nlp.ue_estimater_calibvar import UeEstimatorCalibvar


class UeEstimatorDp:
    def __init__(self, model, dropout_num, reg_or_class, prompt_id):
        self.model = model
        self.dropout_num = dropout_num
        self.reg_or_class = reg_or_class
        self.prompt_id = prompt_id
        
    def __call__(self, dataloader):
        mul_results = self._multi_pred(dataloader)
        pre_calib_results = self._calc_var(mul_results)
        return pre_calib_results
        
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
            mcdp_result['mcdp_var'] = mulvar
        elif self.reg_or_class == 'mix':
            mulscore, mulMP = compute_MixMulMP(mul_results['score'], mul_results['logits'], self.dropout_num, self.prompt_id)
            _, mul_entropy = compute_mulEntropy(mul_results['logits'], self.dropout_num)
            mcdp_result['mcdp_score'] = mulscore
            mcdp_result['mcdp_MP'] = mulMP
        else:
            mulscore, mulMP = compute_mulMP(mul_results['logits'], self.dropout_num)
            _, mul_entropy = compute_mulEntropy(mul_results['logits'], self.dropout_num)
            _, epi_uncertainty = compute_mulprob_epiuncertain(mul_results['logits'], self.dropout_num)
            mcdp_result['mcdp_score'] = mulscore
            mcdp_result['mcdp_MP'] = mulMP
            mcdp_result['mcdp_entropy'] = mul_entropy
            mcdp_result['mcdp_epi_uncertainty'] = epi_uncertainty
        return mcdp_result