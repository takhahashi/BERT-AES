class UeEstimatorTrustscore:
    def __init__(self, model, train_dataset):
        self.model = model
        self.train_dataset = train_dataset
        
    def __call__(self, X, y):
        return self._predict_with_fitted_cov(X, y)
    
    def fit_ue(self, train_dataset):
            
        self._replace_model_head()
        X_features, y = self._exctract_features_and_labels(train_dataset)
            
        self.class_features = self._fit_classfeatures(X_features)
        
    def _fit_classfeatures(self, X_features, y):
        return compute_classfeatures(X_features, y)
    
    
    def _exctract_features(self, dataset):
        model = self.model
        
            
        X_features = cls.predict(X, apply_softmax=False, return_preds=False)[0]
        return X_features

        
    def _predict_with_fitted_cov(self, X, y):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Compute MD with fitted covariance and centroids **************")

        start = time.time()
        if y is None:
            y = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        end = time.time()
        
        eval_results = {}
        
        md, inf_time = mahalanobis_distance(None, None, X_features, 
                                            self.class_cond_centroids, self.class_cond_covariance)
        
        sum_inf_time = inf_time + (end - start)
        eval_results["mahalanobis_distance"] = md.tolist()
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")

        if self.fit_all_md_versions:
            md_relative = mahalanobis_distance_relative(None, None, X_features,
                                                        self.train_centroid, self.train_covariance)

            md_marginal = mahalanobis_distance_marginal(None, None, X_features,
                                                        self.class_cond_centroids, self.class_cond_covariance,
                                                        self.train_centroid, self.train_covariance)

            eval_results["mahalanobis_distance_relative"] = md_relative.tolist()
            eval_results["mahalanobis_distance_marginal"] = md_marginal.tolist()
    
        log.info("**************Done.**********************")
        return eval_results