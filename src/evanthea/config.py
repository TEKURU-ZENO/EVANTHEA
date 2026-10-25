"""Configuration management and typed validation for EVANTHEA experiments."""

from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field


class ProjectPaths(BaseModel):
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    data_splits_dir: Path = Path("data/splits")
    cache_dir: Path = Path("data/interim/cache")
    reports_dir: Path = Path("reports")


class EvantheaConfig(BaseModel):
    project_name: str = "EVANTHEA"
    version: str = "0.1.0"
    random_seed: int = 42
    device: str = "auto"
    paths: ProjectPaths = Field(default_factory=ProjectPaths)
    data: Dict[str, Any] = Field(default_factory=dict)
    models: Dict[str, Any] = Field(default_factory=dict)


def load_config(config_path: Path | str = "configs/base.yaml") -> DictConfig:
    """Loads a Hydra/OmegaConf YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return OmegaConf.load(str(path))
