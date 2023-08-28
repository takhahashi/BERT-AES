from ue4nlp.functions import compute_centroids, compute_covariance, mahalanobis_distance
from models.functions import extract_clsvec_predlabels, extract_clsvec_truelabels
from utils.cfunctions import score_f2int
from utils.dataset import get_score_range
import numpy as np

class UeEstimatorMahalanobis:
    def __init__(self, model, train_dataloader, prompt_id, reg_or_class):
        self.model = model
        self.train_dataloader = train_dataloader
        self.reg_or_class = reg_or_class
        self.prompt_id = prompt_id
        
    def __call__(self, dataloader):
        if self.reg_or_class == 'reg':
            low, high = get_score_range(self.prompt_id)
            X_features, scores = self._extract_features_and_predlabels(dataloader)
            y = np.round(scores * (high - low)).astype('int32')
        else: 
            X_features, scores = self._extract_features_and_predlabels(dataloader)
            y = scores.astype('int32')
        return self._predict_with_fitted_cov(X_features, y)
    
    def fit_ue(self):
        if self.reg_or_class == 'reg':
            low, high = get_score_range(self.prompt_id)
            X_features, y_scaled = self._extract_features_and_truelabels(self.train_dataloader)
            y = np.round(y_scaled * (high - low))
        else:
            X_features, y = self._extract_features_and_truelabels(self.train_dataloader)
            
        self.class_cond_centroids = self._fit_centroids(X_features, y.astype('int32'))
        self.class_cond_covariance = self._fit_covariance(X_features, y.astype('int32'))
        
    def _fit_covariance(self, X, y):
        centroids = self.class_cond_centroids
        return compute_covariance(centroids, X, y)
        
    def _fit_centroids(self, X, y):
        return compute_centroids(X, y)
    
    def _extract_features_and_predlabels(self, data_loader):
        model = self.model
        X_features, predlabels = extract_clsvec_predlabels(model, data_loader)
        return X_features, predlabels
      
    def _extract_features_and_truelabels(self, dataloader):
        model = self.model
        X_features, truelabels = extract_clsvec_truelabels(model, dataloader)
        return X_features, truelabels

    def _predict_with_fitted_cov(self, X_features, y):
        eval_results = {}
        md = mahalanobis_distance(None, 
                                  None, 
                                  X_features, 
                                  self.class_cond_centroids, 
                                  self.class_cond_covariance)
        eval_results["mahalanobis_distance"] = md
        return eval_results