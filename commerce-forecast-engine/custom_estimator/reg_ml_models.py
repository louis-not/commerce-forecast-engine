import numpy as np
from sklearn.base import BaseEstimator
import pandas as pd
from typing import List

class MovingAverageModel(BaseEstimator):
    """
    Sci-kit learn compatible model for Moving Average Model
    """
    target_col: str = ""
    fill_by_feature: bool = False
    y_fitter: List = None

    def __init__(self, window_size, fill_by_feature):
        self.window_size = window_size
        self.fill_by_feature = fill_by_feature

    def fit(self, X: pd.DataFrame, y: pd.Series =None):
        self.target_col = y.name

        if self.fill_by_feature:
            self.check_feature_col_exist(X)
            self.y_fitter = self.return_fitter_from_feature(X.iloc[-1].to_frame().T)
        else:
            self.y_fitter = y

        return self

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X_len = X.shape[0]

        if self.fill_by_feature:
            self.check_feature_col_exist(X)
            y_train = self.return_fitter_from_feature(X.iloc[0].to_frame().T)
        else:
            y_train = self.y_fitter

        y_pred = np.zeros(X_len)
        for i in range(X_len):
            window = y_train[-self.window_size:]
            y_pred[i] = np.mean(window)
            y_train = np.append(y_train, y_pred[i])
        
        return y_pred

    def check_feature_col_exist(self, X: pd.DataFrame):
        feature_col_list = X.columns.values.tolist()
        for i in range(1, self.window_size):
            if f"{self.target_col}_{i}" not in feature_col_list:
                raise ValueError(f"{self.target_col}_{i} not exist in X")
            
    def return_fitter_from_feature(self, X: pd.DataFrame):
        last_row = X.iloc[-1].to_frame().T
        y_fitter = [last_row[f'{self.target_col}_{i}'] for i in range(1, self.window_size + 1)]

        return y_fitter
    

    import numpy as np
from sklearn.base import BaseEstimator
import pandas as pd
from typing import List

class MovingAverageModel(BaseEstimator):
    """
    Sci-kit learn compatible model for Moving Average Model
    """
    target_col: str = ""
    fill_by_feature: bool = False
    y_fitter: List = None

    def __init__(self, window_size, fill_by_feature):
        self.window_size = window_size
        self.fill_by_feature = fill_by_feature

    def fit(self, X: pd.DataFrame, y: pd.Series =None):
        self.target_col = y.name

        if self.fill_by_feature:
            self.check_feature_col_exist(X)
            self.y_fitter = self.return_fitter_from_feature(X.iloc[-1].to_frame().T)
        else:
            self.y_fitter = y

        return self

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X_len = X.shape[0]

        if self.fill_by_feature:
            self.check_feature_col_exist(X)
            y_train = self.return_fitter_from_feature(X.iloc[0].to_frame().T)
        else:
            y_train = self.y_fitter

        y_pred = np.zeros(X_len)
        for i in range(X_len):
            window = y_train[-self.window_size:]
            y_pred[i] = np.mean(window)
            y_train = np.append(y_train, y_pred[i])
        
        return y_pred

    def check_feature_col_exist(self, X: pd.DataFrame):
        feature_col_list = X.columns.values.tolist()
        for i in range(1, self.window_size):
            if f"{self.target_col}_{i}" not in feature_col_list:
                raise ValueError(f"{self.target_col}_{i} not exist in X")
            
    def return_fitter_from_feature(self, X: pd.DataFrame):
        last_row = X.iloc[-1].to_frame().T
        y_fitter = [last_row[f'{self.target_col}_{i}'] for i in range(1, self.window_size + 1)]

        return y_fitter
    


class NaiveBaselineModel(BaseEstimator):
    """
    Sci-kit learn compatible model for Baseline Model
    """
    window_size_avg_sales: float = 0
    last_fit_date: pd.DatetimeIndex = pd.to_datetime("1999-1-1")
    target_col: str = ""
    window_size: int
    look_back_w: int
    fill_by_feature: bool = False

    def __init__(self, look_back_w : int, window_size: int, fill_by_feature):
        """
        Parameter:
        look_back_w: number of weeks to look back before forecasting a new sales
        window_size: window to create average sales
        """
        self.look_back_w = look_back_w
        self.window_size = window_size
        self.fill_by_feature = fill_by_feature
        

    def fit(self, X: pd.DataFrame, y: pd.Series =None):
        self.target_col = y.name
        if self.check_if_look_back_past(X):
            # retrain
            window = y[-self.window_size-self.look_back_w:-self.look_back_w]
            self.window_size_avg_sales = np.mean(window)
            self.last_fit_date = pd.to_datetime(X.index.max())
        else:
            # not retrain
            pass
        return self

    def predict(self, X):
        X = X.copy()
        X_len = X.shape[0]
        
        if self.check_if_look_back_past(X):
            if self.fill_by_feature:
                self.check_feature_col_exist(X)
                window = self.return_fitter_from_feature(X)[-self.window_size:]
                self.window_size_avg_sales = np.mean(window)
                self.last_fit_date = X.index.min()
            else:
                raise KeyError(f"Fill by non feature haven't been developed")
        else:
            pass

        y_pred = np.ones(X_len) * self.window_size_avg_sales

        return y_pred


    def check_if_look_back_past(self, X):
        min_index = X.index.min()
        days_past = min_index - self.last_fit_date
        if days_past.days >= self.look_back_w * 7 :
            return True
        return False

    def check_feature_col_exist(self, X: pd.DataFrame):
        feature_col_list = X.columns.values.tolist()
        for i in range(1, self.window_size):
            if f"{self.target_col}_{i}" not in feature_col_list:
                raise ValueError(f"{self.target_col}_{i} not exist in X")
            
    def return_fitter_from_feature(self, X: pd.DataFrame):
        last_row = X.iloc[0].to_frame().T
        y_fitter = [last_row[f'{self.target_col}_{i}'] for i in range(1, self.window_size + self.look_back_w + 1)]
        return y_fitter
