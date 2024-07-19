import pandas as pd
from dataclasses import dataclass
from typing import Dict
import numpy as np
from sklearn.base import clone

from source.forecast_engine.models.model_config import ModelConfig
from source.forecast_engine.feature import FeatureConfigV4
from source.logger import create_logger

logger = create_logger(__name__)

@dataclass
class MLForecast:
    def __init__(
        self, model_config: ModelConfig = None, path: str = None, fit_kwargs: Dict = {}
    ) -> None:
        """
        Convenient wrapper around scikit-learn style estimators
        """
        if model_config is None:
            self.load_model_config(path)
        else:
            self.model_config = model_config

        self._model = clone(self.model_config.model)
        

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        fit_kwargs: Dict = {},
    ):
        """
        for training model

        Parameters:
            - X: Features
            - y: target
        """
        # convert X and y
        self._train_features = X.columns.tolist()
        self._train_targets = y.values.tolist()

        # modeling
        self._model.fit(X, y, **fit_kwargs)

        return self

    def predict(self, X: pd.DataFrame, feat_config: FeatureConfigV4 = None) -> pd.Series:
        """
        Predicts on the given dataframe using the trained model
        Documentation later

        Parameter:
        X : Features

        Return:
        y : predicted (pd.Series)
        """
        if not feat_config:
            logger.warning("Using whole forecasting technique might be not accurate for iterative features")
            return pd.Series(
                self._model.predict(X).ravel(),
                index=X.index,
                name=f"{self.model_config.name}",
            )

        else:
            y_preds = []
            y_pred = None
            for i in range(len(X)):
                X_row = X.iloc[i:i+1]
                index = X_row.index

                # Update X_row dynamic value from previous y_pred
                if y_pred:
                    X_update = feat_config.update_dynamic_features(X, pd.Series(y_pred,index))
                    X_row = X_update.loc[index]

                y_pred = self._model.predict(X_row)[0]
                y_preds.append(y_pred)

            return pd.Series(
                        y_preds,
                        index=X.index,
                        name=f"{self.model_config.name}",
                    )
