"""EVANTHEA Fitness Prediction Engine: Supervised surrogates, feature extraction, and baselines."""

from evanthea.predictors.base import FitnessPredictor
from evanthea.predictors.baselines import (
    MeanPredictor,
    MedianPredictor,
    TrueZeroShotESM2Predictor,
    ZeroShotESM2Predictor,
)
from evanthea.predictors.e2e_plm import E2ELoRALinear, EndToEndPLMPredictor
from evanthea.predictors.gradient_boosting import HistGradientBoostingPredictor
from evanthea.predictors.plm_finetune import (
    AttentiveDeepRegressionHead,
    CompositeRankLoss,
    LoRALinear,
    PLMFineTuningPredictor,
)
from evanthea.predictors.random_forest import RandomForestFitnessPredictor
from evanthea.predictors.ridge import RidgeFitnessPredictor
from evanthea.predictors.targets import TargetTransformer

__all__ = [
    "FitnessPredictor",
    "TargetTransformer",
    "RidgeFitnessPredictor",
    "HistGradientBoostingPredictor",
    "RandomForestFitnessPredictor",
    "MeanPredictor",
    "MedianPredictor",
    "ZeroShotESM2Predictor",
    "TrueZeroShotESM2Predictor",
    "PLMFineTuningPredictor",
    "LoRALinear",
    "AttentiveDeepRegressionHead",
    "CompositeRankLoss",
    "EndToEndPLMPredictor",
    "E2ELoRALinear",
]
