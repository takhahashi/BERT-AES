from models.functions import return_predresults, extract_clsvec_labels
from ue4nlp.functions import sep_features_by_class, diffclass_euclid_dist, sameclass_euclid_dist
from utils.cfunctions import score_f2int
import numpy as np

class UeEstimatorTrustscore:
    def __init__(self, model, train_dataloader, prompt_id):
        self.model = model
        self.train_dataloader = train_dataloader
        self.prompt_id = prompt_id
        
    def __call__(self, dataloader=None, X_features=None, scores=None):
        if X_features is not None and scores is not None:
            return self._predict_with_fitted_clsvec(X_features, scores)
        else:
            X_features, scores = self._exctract_features_and_labels(dataloader)
            int_scores = score_f2int(scores, self.prompt_id)
            return self._predict_with_fitted_clsvec(X_features, int_scores)
    
    def fit_ue(self):
        X_features, y = self._exctract_features_and_labels(self.train_dataloader)
        int_labels = score_f2int(y, self.prompt_id)
        self.class_features = self._fit_classfeatures(X_features, int_labels)
        
    def _fit_classfeatures(self, X_features, scores):
        return sep_features_by_class(X_features, scores)
    
    def _exctract_features_and_labels(self, data_loader):
        model = self.model
        results = extract_clsvec_labels(model, data_loader)
        return results['hidden_state'], results['labels']

        
    def _predict_with_fitted_cov(self, X_features, labels):
        trust_score_values = []
        
        for x_feature, label in zip(X_features, labels):
            diffclass_dist = diffclass_euclid_dist(x_feature, label, self.class_features)
            sameclass_dist= sameclass_euclid_dist(x_feature, label, self.class_features)
            if sameclass_dist is None:
                trust_score_values = np.append(trust_score_values, 0.)
            else:
                trust_score = diffclass_dist / (diffclass_dist + sameclass_dist)
                trust_score_values = np.append(trust_score_values, trust_score)
        eval_results = {'trust_score': trust_score_values}
        return eval_results