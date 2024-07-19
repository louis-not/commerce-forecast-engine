from sklearn.base import BaseEstimator
import numpy as np

class CrossProductLearning(BaseEstimator):
    """
    Sci-kit learn compatible model for Cold Start Model
    """
    
    config_type = "CROSS_PRODUCT_LEARNING"

    def __init__(self, cluster_strategy:str) -> None:
        self.cluster_strategy = cluster_strategy

    def fit(self,X,y):
        pass

    def predict(self,X):
        pass