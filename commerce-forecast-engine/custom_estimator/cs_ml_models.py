from sklearn.base import BaseEstimator
import numpy as np

class AverageEarlySales(BaseEstimator):
    """
    Sci-kit learn compatible model for Cold Start Model
    """

    ref_sales = np.zeros(0)

    config_type = "AVERAGE_EARLY_SALES"

    def __init__(self, cluster_strategy: str) -> None:
        self.cluster_strategy = cluster_strategy

    def fit(self,X,y):
        self.ref_sales = min(y[0], self.cap_value)

    def predict(self,X):
        return np.ones(X.shape[0]) * self.ref_sales