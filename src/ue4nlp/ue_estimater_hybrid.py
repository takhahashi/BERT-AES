class UeEstimatorHybrid:
    def __init__(self, aleatric_ue, epistemic_ue, ):
        self.ale_u = aleatric_ue
        self.epi_u = epistemic_ue
    
    def __call__(self)