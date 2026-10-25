"""Unit tests for configuration loading and validation."""

from pathlib import Path
import pytest
from evanthea.config import EvantheaConfig, load_config


def test_load_config_base():
    cfg = load_config("configs/base.yaml")
    assert cfg.project.name == "EVANTHEA"
    assert cfg.project.version == "0.1.0"
    assert cfg.project.random_seed == 42


def test_evanthea_config_model():
    model = EvantheaConfig()
    assert model.project_name == "EVANTHEA"
    assert model.paths.data_raw_dir == Path("data/raw")


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent.yaml")
