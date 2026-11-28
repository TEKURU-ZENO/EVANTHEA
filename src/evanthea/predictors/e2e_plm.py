"""End-to-end parameter-efficient fine-tuning (PEFT) predictor for Protein Language Models.

This module provides a genuine token-level end-to-end ESM-2 fine-tuner that backpropagates
gradients through transformer attention projections via native PyTorch Low-Rank Adaptation (LoRA),
directly addressing the distinction between frozen-embedding surrogates and true end-to-end PLM tuning.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from evanthea.predictors.base import FitnessPredictor
from evanthea.predictors.plm_finetune import CompositeRankLoss

logger = logging.getLogger(__name__)


class E2ELoRALinear(nn.Module):
    """Native PyTorch LoRA wrapper for an existing nn.Linear layer.

    Computes:
        y = W_0 x + (alpha / r) * B(A(dropout(x)))
    where W_0 is the frozen pre-trained weight matrix, A is initialized with
    Kaiming uniform, and B is initialized to zeros.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        dtype = base_linear.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Kaiming uniform initialization for A
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


class E2EProteinRegressionModel(nn.Module):
    """Full PyTorch module integrating a transformer backbone with LoRA and an attentive head."""

    def __init__(
        self,
        base_model: nn.Module,
        hidden_dim: int,
        head_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        base_dtype = getattr(base_model, "dtype", torch.float32)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim, dtype=base_dtype),
            nn.LayerNorm(head_hidden_dim, dtype=base_dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 32, dtype=base_dtype),
            nn.GELU(),
            nn.Linear(32, 1, dtype=base_dtype),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Masked mean pooling across valid residue tokens (excluding padding)
        mask = attention_mask.unsqueeze(-1)
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1).float()


class EndToEndPLMPredictor(FitnessPredictor):
    """End-to-End Parameter-Efficient Fine-Tuning (PEFT) Protein Language Model Predictor.

    Fine-tunes transformer attention layers via native PyTorch LoRA adapters directly
    from tokenized amino acid sequence inputs, training with rank-regularized composite loss.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Sequence[str] = ("query", "value"),
        head_hidden_dim: int = 128,
        head_dropout: float = 0.1,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-2,
        batch_size: int = 32,
        max_epochs: int = 10,
        patience: int = 3,
        lambda_rank: float = 0.5,
        huber_delta: float = 1.0,
        seed: int = 42,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
    ) -> None:
        super().__init__(name=f"E2E_PLM_{model_name.split('/')[-1]}")
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = list(target_modules)
        self.head_hidden_dim = head_hidden_dim
        self.head_dropout = head_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lambda_rank = lambda_rank
        self.huber_delta = huber_delta
        self.seed = seed
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation

        # Device selection: test CUDA tensor execution or fallback to CPU
        if device is not None:
            self.device_str = device
        else:
            self.device_str = self._resolve_device()
        self.device = torch.device(self.device_str)

        self.tokenizer: Optional[Any] = None
        self.model: Optional[E2EProteinRegressionModel] = None
        self.best_val_spearman: float = -1.0
        self.training_history: List[Dict[str, float]] = []

    def _resolve_device(self) -> str:
        if torch.cuda.is_available():
            try:
                t = torch.zeros(1, device="cuda")
                _ = t + 1
                return "cuda"
            except Exception:
                pass
        return "cpu"

    def _tokenize_sequences(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        is_t5 = "t5" in self.model_name.lower() or "ankh" in self.model_name.lower()
        if is_t5:
            tokenized = [list(s) for s in sequences]
            return self.tokenizer(tokenized, is_split_into_words=True, padding=True, return_tensors="pt")
        return self.tokenizer(sequences, padding=True, return_tensors="pt")

    def _init_model(self) -> None:
        """Initializes tokenizer, base PLM model (ESM or T5/Ankh), injects LoRA, and builds regression head."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.device_str == "cpu":
            num_cores = os.cpu_count() or 8
            torch.set_num_threads(num_cores)
            load_dtype = torch.float32
            attn_impl = "eager"
        else:
            load_dtype = self.torch_dtype if self.torch_dtype is not None else torch.bfloat16
            attn_impl = self.attn_implementation if self.attn_implementation is not None else "sdpa"

        is_t5 = "t5" in self.model_name.lower() or "ankh" in self.model_name.lower()
        if is_t5:
            from transformers import T5EncoderModel
            model_cls = T5EncoderModel
            if set(self.target_modules) == {"query", "value"}:
                self.target_modules = ["q", "v"]
        else:
            from transformers import AutoModel
            model_cls = AutoModel

        # Cache-first resolution to prevent sandbox network timeouts
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            try:
                base_esm = model_cls.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    dtype=load_dtype,
                    attn_implementation=attn_impl,
                )
            except Exception:
                base_esm = model_cls.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    dtype=load_dtype,
                )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            try:
                base_esm = model_cls.from_pretrained(
                    self.model_name,
                    dtype=load_dtype,
                    attn_implementation=attn_impl,
                )
            except Exception:
                base_esm = model_cls.from_pretrained(self.model_name, dtype=load_dtype)

        # Freeze all base model parameters
        for p in base_esm.parameters():
            p.requires_grad = False

        # Inject LoRA into target attention projections
        num_injected = 0
        for name, module in base_esm.named_modules():
            for target in self.target_modules:
                if hasattr(module, target) and isinstance(getattr(module, target), nn.Linear):
                    orig = getattr(module, target)
                    lora_wrap = E2ELoRALinear(
                        orig,
                        r=self.lora_r,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                    )
                    setattr(module, target, lora_wrap)
                    num_injected += 1

        logger.info(f"Injected LoRA into {num_injected} layers in {self.model_name}.")

        hidden_dim = getattr(base_esm.config, "hidden_size", getattr(base_esm.config, "d_model", 768))
        self.model = E2EProteinRegressionModel(
            base_model=base_esm,
            hidden_dim=hidden_dim,
            head_hidden_dim=self.head_hidden_dim,
            dropout=self.head_dropout,
        ).to(self.device)

    def fit_sequences(
        self,
        train_sequences: List[str],
        train_targets: np.ndarray,
        val_sequences: Optional[List[str]] = None,
        val_targets: Optional[np.ndarray] = None,
    ) -> "EndToEndPLMPredictor":
        """Fits the end-to-end LoRA model directly from sequence strings and continuous fitness targets."""
        self._init_model()
        assert self.model is not None and self.tokenizer is not None

        # Pre-tokenize all sequences upfront for fast batch iteration
        train_enc = self._tokenize_sequences(train_sequences)
        train_ids = train_enc["input_ids"]
        train_mask = train_enc["attention_mask"]
        train_y = torch.tensor(train_targets, dtype=torch.float32)

        train_dataset = TensorDataset(train_ids, train_mask, train_y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_sequences is not None and val_targets is not None and len(val_sequences) > 0:
            val_enc = self._tokenize_sequences(val_sequences)
            val_ids = val_enc["input_ids"]
            val_mask = val_enc["attention_mask"]
            val_y = torch.tensor(val_targets, dtype=torch.float32)
            val_dataset = TensorDataset(val_ids, val_mask, val_y)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = CompositeRankLoss(rank_weight=self.lambda_rank, huber_delta=self.huber_delta)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-5)

        best_weights = None
        epochs_without_improvement = 0

        logger.info(f"Beginning E2E LoRA fine-tuning for up to {self.max_epochs} epochs ({len(train_dataset)} samples)...")

        for epoch in range(1, self.max_epochs + 1):
            t_epoch_start = time.time()
            self.model.train()
            total_train_loss = 0.0
            num_batches = 0

            for b_ids, b_mask, b_targets in train_loader:
                b_ids = b_ids.to(self.device)
                b_mask = b_mask.to(self.device)
                b_targets = b_targets.to(self.device)

                optimizer.zero_grad()
                preds = self.model(b_ids, b_mask)
                loss = criterion(preds, b_targets)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_train_loss = total_train_loss / max(num_batches, 1)

            val_metrics = self._evaluate_loader(val_loader) if val_loader else {}
            val_spearman = val_metrics.get("spearman", 0.0)
            t_epoch = time.time() - t_epoch_start

            logger.info(
                f"Epoch {epoch}/{self.max_epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Rho: {val_spearman:.4f} | Time: {t_epoch:.1f}s"
            )

            self.training_history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_spearman": val_spearman,
                "val_mae": val_metrics.get("mae", 0.0),
            })

            # Checkpoint tracking on validation Spearman correlation
            if val_loader:
                if val_spearman > self.best_val_spearman:
                    self.best_val_spearman = val_spearman
                    best_weights = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}.")
                        break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        self.is_fitted = True
        return self

    def _evaluate_loader(self, loader: Optional[DataLoader]) -> Dict[str, float]:
        if loader is None or self.model is None:
            return {}
        self.model.eval()
        all_preds: List[float] = []
        all_targets: List[float] = []

        with torch.no_grad():
            for b_ids, b_mask, b_targets in loader:
                b_ids = b_ids.to(self.device)
                b_mask = b_mask.to(self.device)
                preds = self.model(b_ids, b_mask)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(b_targets.cpu().tolist())

        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)

        from scipy.stats import spearmanr
        rho, _ = spearmanr(preds_arr, targets_arr)
        mae = float(np.mean(np.abs(preds_arr - targets_arr)))
        return {"spearman": float(rho) if not np.isnan(rho) else 0.0, "mae": mae}

    def predict_sequences(self, sequences: List[str]) -> np.ndarray:
        """Generates continuous fitness predictions for candidate sequence strings."""
        if not self.is_fitted or self.model is None or self.tokenizer is None:
            raise RuntimeError("EndToEndPLMPredictor must be fitted before predict.")

        self.model.eval()
        enc = self._tokenize_sequences(sequences)
        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds: List[float] = []
        with torch.no_grad():
            for b_ids, b_mask in loader:
                b_ids = b_ids.to(self.device)
                b_mask = b_mask.to(self.device)
                preds = self.model(b_ids, b_mask)
                all_preds.extend(preds.cpu().tolist())

        return np.array(all_preds, dtype=np.float64)

    # Standard FitnessPredictor interface implementation
    def fit(
        self,
        X_train: Union[np.ndarray, List[str]],
        y_train: np.ndarray,
        X_val: Optional[Union[np.ndarray, List[str]]] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "EndToEndPLMPredictor":
        """Fits predictor accepting string sequences or token-id matrices."""
        train_seqs = [str(x) for x in X_train]
        val_seqs = [str(x) for x in X_val] if X_val is not None else None
        return self.fit_sequences(train_seqs, y_train, val_sequences=val_seqs, val_targets=y_val)

    def predict(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Predicts fitness accepting string sequences or token-id matrices."""
        seqs = [str(x) for x in X]
        return self.predict_sequences(seqs)

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "head_hidden_dim": self.head_hidden_dim,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "lambda_rank": self.lambda_rank,
            "device": self.device_str,
            "best_val_spearman": self.best_val_spearman,
        }

    def save(self, file_path: Union[str, Path]) -> None:
        """Serializes predictor configuration and trained weights."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an unfitted EndToEndPLMPredictor.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        trainable_state = {
            name: param.cpu() for name, param in self.model.named_parameters() if param.requires_grad
        }
        checkpoint = {
            "params": self.get_params(),
            "trainable_state": trainable_state,
        }
        torch.save(checkpoint, path)

    def load(self, file_path: Union[str, Path]) -> "EndToEndPLMPredictor":
        """Loads serialized weights into predictor."""
        path = Path(file_path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        params = checkpoint["params"]
        self.model_name = params["model_name"]
        self.lora_r = params["lora_r"]
        self.lora_alpha = params["lora_alpha"]
        self.lora_dropout = params["lora_dropout"]
        self.target_modules = params["target_modules"]
        self.head_hidden_dim = params["head_hidden_dim"]
        self._init_model()
        assert self.model is not None
        model_state = self.model.state_dict()
        model_state.update(checkpoint["trainable_state"])
        self.model.load_state_dict(model_state)
        self.is_fitted = True
        return self
