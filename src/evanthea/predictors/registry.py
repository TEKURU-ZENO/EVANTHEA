"""Model registry for versioning, persisting, and reloading trained fitness predictors and scalers."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib

from evanthea.evaluation.metrics import MetricBundle
from evanthea.predictors.base import FitnessPredictor
from evanthea.predictors.targets import TargetTransformer


class ModelRegistry:
    """Manages serialization, versioning, and provenance of trained surrogate models."""

    def __init__(self, registry_dir: Union[Path, str] = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: FitnessPredictor,
        target_transformer: TargetTransformer,
        feature_config: str,
        metrics: MetricBundle,
        model_id: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Serializes model weights, target scalers, and provenance metadata."""
        model_path = self.registry_dir / model_id
        model_path.mkdir(parents=True, exist_ok=True)

        # 1. Save model weights
        joblib.dump(model, model_path / "model.joblib")

        # 2. Save TargetTransformer
        with open(model_path / "target_transformer.json", "w", encoding="utf-8") as f:
            json.dump(target_transformer.get_params(), f, indent=2)

        # 3. Save evaluation metrics and provenance
        meta: Dict[str, Any] = {
            "model_id": model_id,
            "model_name": model.name,
            "feature_config": feature_config,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hyperparameters": model.get_params(),
            "metrics": metrics.model_dump(),
            "extra_metadata": extra_metadata or {},
        }
        with open(model_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return model_path

    def load_model(
        self,
        model_id: str,
    ) -> Tuple[FitnessPredictor, TargetTransformer, Dict[str, Any]]:
        """Loads a persisted model, target transformer, and metadata."""
        model_path = self.registry_dir / model_id
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model ID '{model_id}' not found in registry at {model_path}.")

        model: FitnessPredictor = joblib.load(model_path / "model.joblib")

        with open(model_path / "target_transformer.json", "r", encoding="utf-8") as f:
            tf_data = json.load(f)

        transformer = TargetTransformer(mode=tf_data["mode"])
        transformer.mean_ = tf_data.get("mean")
        transformer.scale_ = tf_data.get("scale")
        transformer.is_fitted = tf_data.get("is_fitted", True)
        transformer.n_samples_seen_ = tf_data.get("n_samples_seen", 0)

        with open(model_path / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        return model, transformer, meta

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns metadata for all models currently registered."""
        models = []
        for p in self.registry_dir.iterdir():
            if p.is_dir() and (p / "metadata.json").is_file():
                try:
                    with open(p / "metadata.json", "r", encoding="utf-8") as f:
                        models.append(json.load(f))
                except Exception:
                    pass
        return models
