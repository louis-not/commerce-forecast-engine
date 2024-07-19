"""
Standarized code to train and evaluate ML model
"""
from dataclasses import MISSING, field, dataclass
from sklearn.base import BaseEstimator, clone


@dataclass
class ModelConfig:
    model: BaseEstimator = field(
        default=MISSING, metadata={"help": "Sci-kit learn compatible model instance"}
    )

    name: str = field(
        default=None,
        metadata={"help": "Name or identifier for the model"},
    )
    
    is_one_step_forecast: bool = field(
        default=False,
        metadata={"help": "True if model constrained to forecast one-step after the model trained"},
    )

    lagging_tresshold: int = field(
        default=0,
        metadata={"help": "Number of lagged features needed to be available for the Model to run"},
    )

    def clone(self):
        self.model = clone(self.model)
        return self
