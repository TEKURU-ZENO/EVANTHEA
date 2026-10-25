"""Abstract Protein Language Model interface and Hugging Face Transformers ESM-2 implementation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import torch
from pydantic import BaseModel, ConfigDict, Field

from evanthea.constants import (
    DEFAULT_ESM2_150M_CHECKPOINT,
    ESM2_MAX_SEQUENCE_LENGTH,
    STANDARD_AMINO_ACIDS,
)
from evanthea.data.schemas import Mutation
from evanthea.exceptions import CoordinateOutOfBoundsError, ModelInferenceError, SequenceMismatchError
from evanthea.protein.sequence import apply_mutations, validate_amino_acid_string


class SequenceEmbeddingResult(BaseModel):
    """Container for sequence-level and residue-level biological representations."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sequence_embeddings: torch.Tensor = Field(..., description="Mean-pooled sequence embeddings of shape (B, D)")
    residue_embeddings: List[torch.Tensor] = Field(..., description="Unpadded per-residue embeddings of shape (L_i, D)")
    hidden_dim: int = Field(..., description="Embedding dimension D")
    model_name: str = Field(..., description="Checkpoint name that generated representations")


class MutationScoreResult(BaseModel):
    """Scoring result for a single amino acid substitution."""
    mutation: Mutation
    log_odds_score: float = Field(..., description="log p(mut | context) - log p(wt | context)")
    wt_log_prob: float = Field(..., description="log probability of wildtype residue")
    mut_log_prob: float = Field(..., description="log probability of substituted residue")
    scoring_method: str = Field(..., description="'masked_marginal' or 'wildtype_marginal'")


class CombinatorialScoreResult(BaseModel):
    """Scoring result for combinatorial multi-mutant variants."""
    mutations: List[Mutation]
    variant_sequence: str
    score: float = Field(..., description="Composite log-odds score")
    additive_score: float = Field(..., description="Sum of individual point mutation scores")
    scoring_method: str = Field(..., description="'additive' or 'joint_masked'")


class ProteinLanguageModel(ABC):
    """Abstract interface decoupling downstream evolution algorithms from the underlying PLM backend."""

    @abstractmethod
    def embed(
        self,
        sequences: List[str],
        batch_size: int = 16,
    ) -> SequenceEmbeddingResult:
        """Extracts sequence-level and per-residue representations.

        Args:
            sequences: List of canonical amino acid strings.
            batch_size: Batch size for forward passes.

        Returns:
            SequenceEmbeddingResult with pooled and token-level tensors.
        """
        raise NotImplementedError

    @abstractmethod
    def score_mutations(
        self,
        sequence: str,
        mutations: List[Mutation],
        method: Literal["masked_marginal", "wildtype_marginal"] = "masked_marginal",
    ) -> List[MutationScoreResult]:
        """Calculates zero-shot log-odds scores for point mutations against a reference sequence.

        Args:
            sequence: Wild-type reference sequence.
            mutations: Point mutations to evaluate.
            method: 'masked_marginal' (exact mask) or 'wildtype_marginal' (fast unmasked).
        """
        raise NotImplementedError

    @abstractmethod
    def score_combinatorial_variant(
        self,
        sequence: str,
        mutations: List[Mutation],
        method: Literal["additive", "joint_masked"] = "additive",
    ) -> CombinatorialScoreResult:
        """Scores a multi-mutant combination."""
        raise NotImplementedError

    @abstractmethod
    def score_sequence_plausibility(self, sequence: str) -> float:
        """Computes sequence pseudo-perplexity (PLL). Lower value indicates higher naturalness."""
        raise NotImplementedError

    @abstractmethod
    def get_masked_marginal_logits(
        self,
        sequence: str,
        position_1based: int,
    ) -> Dict[str, float]:
        """Returns logits for all 20 canonical amino acids when position is masked."""
        raise NotImplementedError

    @abstractmethod
    def get_wildtype_marginal_logits(
        self,
        sequence: str,
    ) -> Dict[int, Dict[str, float]]:
        """Returns logits for all 20 canonical amino acids across all sequence positions."""
        raise NotImplementedError


class HuggingFaceESM2Model(ProteinLanguageModel):
    """Production implementation of ESM-2 using Hugging Face Transformers."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_ESM2_150M_CHECKPOINT,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float16,
    ):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.checkpoint = checkpoint
        self.requested_dtype = dtype

        # Device determination: support CUDA if available and functional, else CPU
        if device is None:
            if torch.cuda.is_available():
                try:
                    # Test if CUDA execution actually works (guards against sm_120 driver/kernel incompatibilities)
                    t_test = torch.zeros(1, device="cuda")
                    del t_test
                    self.device = torch.device("cuda")
                except RuntimeError:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Force float32 if running on CPU
        if self.device.type == "cpu":
            self.dtype = torch.float32
        else:
            self.dtype = dtype

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.model = AutoModelForMaskedLM.from_pretrained(checkpoint, torch_dtype=self.dtype)
            self.model.to(self.device)
            self.model.eval()

            # Rule 1: Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

        except Exception as e:
            raise ModelInferenceError(f"Failed to load ESM-2 model from '{checkpoint}': {e}") from e

        self.hidden_dim = self.model.config.hidden_size
        self.mask_token_id = self.tokenizer.mask_token_id

    @torch.inference_mode()
    def embed(
        self,
        sequences: List[str],
        batch_size: int = 16,
    ) -> SequenceEmbeddingResult:
        """Extracts representations with dynamic length bucketing."""
        if not sequences:
            raise ValueError("Sequence list cannot be empty.")

        cleaned_seqs = [validate_amino_acid_string(s) for s in sequences]

        # Check lengths
        for s in cleaned_seqs:
            if len(s) > ESM2_MAX_SEQUENCE_LENGTH:
                raise CoordinateOutOfBoundsError(
                    len(s), ESM2_MAX_SEQUENCE_LENGTH, f"Sequence length exceeds ESM-2 max limit {ESM2_MAX_SEQUENCE_LENGTH}"
                )

        pooled_list: List[torch.Tensor] = []
        residue_list: List[torch.Tensor] = []

        for i in range(0, len(cleaned_seqs), batch_size):
            batch = cleaned_seqs[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=True,
            ).to(self.device)

            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]  # (B, Seq_Len_with_special, D)

            for b_idx, seq in enumerate(batch):
                seq_len = len(seq)
                # Position 1 to seq_len corresponds to actual residues (skipping <cls> at index 0)
                res_emb = last_hidden_state[b_idx, 1 : 1 + seq_len, :].detach().cpu()
                mean_emb = res_emb.mean(dim=0)  # (D,)

                pooled_list.append(mean_emb)
                residue_list.append(res_emb)

        stacked_pooled = torch.stack(pooled_list, dim=0)
        return SequenceEmbeddingResult(
            sequence_embeddings=stacked_pooled,
            residue_embeddings=residue_list,
            hidden_dim=self.hidden_dim,
            model_name=self.checkpoint,
        )

    @torch.inference_mode()
    def score_mutations(
        self,
        sequence: str,
        mutations: List[Mutation],
        method: Literal["masked_marginal", "wildtype_marginal"] = "masked_marginal",
    ) -> List[MutationScoreResult]:
        """Calculates zero-shot log-odds scores."""
        ref_seq = validate_amino_acid_string(sequence)
        seq_len = len(ref_seq)

        # Validate coordinates and wildtype residues
        for mut in mutations:
            if mut.position_1based < 1 or mut.position_1based > seq_len:
                raise CoordinateOutOfBoundsError(mut.position_1based, seq_len, mut.to_string())
            actual_wt = ref_seq[mut.position_0based]
            if actual_wt != mut.ref_residue:
                raise SequenceMismatchError(mut.position_1based, mut.ref_residue, actual_wt, mut.to_string())

        results: List[MutationScoreResult] = []

        if method == "wildtype_marginal":
            # Fast O(1) pass on unmasked wild-type sequence
            inputs = self.tokenizer(ref_seq, return_tensors="pt", add_special_tokens=True).to(self.device)
            logits = self.model(**inputs).logits[0]  # (Seq_Len+2, Vocab)
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            for mut in mutations:
                tok_idx = mut.position_1based  # +1 offset for <cls>
                wt_id = self.tokenizer.convert_tokens_to_ids(mut.ref_residue)
                mut_id = self.tokenizer.convert_tokens_to_ids(mut.mut_residue)

                wt_lp = log_probs[tok_idx, wt_id].item()
                mut_lp = log_probs[tok_idx, mut_id].item()
                log_odds = mut_lp - wt_lp

                results.append(
                    MutationScoreResult(
                        mutation=mut,
                        log_odds_score=log_odds,
                        wt_log_prob=wt_lp,
                        mut_log_prob=mut_lp,
                        scoring_method=method,
                    )
                )

        elif method == "masked_marginal":
            # Mask each mutation position individually
            ref_chars = list(ref_seq)
            for mut in mutations:
                masked_chars = list(ref_chars)
                masked_chars[mut.position_0based] = self.tokenizer.mask_token
                masked_str = "".join(masked_chars)

                inputs = self.tokenizer(masked_str, return_tensors="pt", add_special_tokens=True).to(self.device)
                logits = self.model(**inputs).logits[0]
                log_probs = torch.log_softmax(logits.float(), dim=-1)

                tok_idx = mut.position_1based
                wt_id = self.tokenizer.convert_tokens_to_ids(mut.ref_residue)
                mut_id = self.tokenizer.convert_tokens_to_ids(mut.mut_residue)

                wt_lp = log_probs[tok_idx, wt_id].item()
                mut_lp = log_probs[tok_idx, mut_id].item()
                log_odds = mut_lp - wt_lp

                results.append(
                    MutationScoreResult(
                        mutation=mut,
                        log_odds_score=log_odds,
                        wt_log_prob=wt_lp,
                        mut_log_prob=mut_lp,
                        scoring_method=method,
                    )
                )

        return results

    @torch.inference_mode()
    def score_combinatorial_variant(
        self,
        sequence: str,
        mutations: List[Mutation],
        method: Literal["additive", "joint_masked"] = "additive",
    ) -> CombinatorialScoreResult:
        """Scores a multi-mutant combination."""
        ref_seq = validate_amino_acid_string(sequence)
        mutated_seq = apply_mutations(ref_seq, mutations)

        if not mutations:
            return CombinatorialScoreResult(
                mutations=[],
                variant_sequence=ref_seq,
                score=0.0,
                additive_score=0.0,
                scoring_method=method,
            )

        # Single mutation scores
        single_scores = self.score_mutations(ref_seq, mutations, method="masked_marginal")
        additive_sum = sum(s.log_odds_score for s in single_scores)

        if method == "additive":
            return CombinatorialScoreResult(
                mutations=mutations,
                variant_sequence=mutated_seq,
                score=additive_sum,
                additive_score=additive_sum,
                scoring_method="additive",
            )

        elif method == "joint_masked":
            # Mask all mutated positions simultaneously
            seq_chars = list(ref_seq)
            for m in mutations:
                seq_chars[m.position_0based] = self.tokenizer.mask_token
            masked_str = "".join(seq_chars)

            inputs = self.tokenizer(masked_str, return_tensors="pt", add_special_tokens=True).to(self.device)
            logits = self.model(**inputs).logits[0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            joint_sum = 0.0
            for m in mutations:
                tok_idx = m.position_1based
                wt_id = self.tokenizer.convert_tokens_to_ids(m.ref_residue)
                mut_id = self.tokenizer.convert_tokens_to_ids(m.mut_residue)
                wt_lp = log_probs[tok_idx, wt_id].item()
                mut_lp = log_probs[tok_idx, mut_id].item()
                joint_sum += (mut_lp - wt_lp)

            return CombinatorialScoreResult(
                mutations=mutations,
                variant_sequence=mutated_seq,
                score=joint_sum,
                additive_score=additive_sum,
                scoring_method="joint_masked",
            )

        raise ValueError(f"Unknown scoring method: {method}")

    @torch.inference_mode()
    def score_sequence_plausibility(self, sequence: str) -> float:
        r"""Computes pseudo-perplexity (PLL). Lower is more natural/plausible.

        Formula: PLL = exp( - 1/L * sum_i log p(x_i | x_\i) )
        """
        ref_seq = validate_amino_acid_string(sequence)
        seq_len = len(ref_seq)
        seq_chars = list(ref_seq)

        total_log_likelihood = 0.0

        for idx, char in enumerate(seq_chars):
            masked = list(seq_chars)
            masked[idx] = self.tokenizer.mask_token
            masked_str = "".join(masked)

            inputs = self.tokenizer(masked_str, return_tensors="pt", add_special_tokens=True).to(self.device)
            logits = self.model(**inputs).logits[0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            tok_idx = idx + 1
            char_id = self.tokenizer.convert_tokens_to_ids(char)
            total_log_likelihood += log_probs[tok_idx, char_id].item()

        avg_nll = - (total_log_likelihood / seq_len)
        pll = torch.exp(torch.tensor(avg_nll)).item()
        return pll

    @torch.inference_mode()
    def get_masked_marginal_logits(
        self,
        sequence: str,
        position_1based: int,
    ) -> Dict[str, float]:
        """Returns logits for all 20 canonical amino acids when position is masked."""
        ref_seq = validate_amino_acid_string(sequence)
        seq_len = len(ref_seq)
        if position_1based < 1 or position_1based > seq_len:
            raise CoordinateOutOfBoundsError(position_1based, seq_len, f"pos={position_1based}")

        masked_chars = list(ref_seq)
        masked_chars[position_1based - 1] = self.tokenizer.mask_token
        masked_str = "".join(masked_chars)

        inputs = self.tokenizer(masked_str, return_tensors="pt", add_special_tokens=True).to(self.device)
        logits = self.model(**inputs).logits[0]
        tok_idx = position_1based

        res: Dict[str, float] = {}
        for aa in sorted(STANDARD_AMINO_ACIDS):
            tok_id = self.tokenizer.convert_tokens_to_ids(aa)
            res[aa] = logits[tok_idx, tok_id].item()
        return res

    @torch.inference_mode()
    def get_wildtype_marginal_logits(
        self,
        sequence: str,
    ) -> Dict[int, Dict[str, float]]:
        """Returns logits for all 20 canonical amino acids across all sequence positions in a single pass."""
        ref_seq = validate_amino_acid_string(sequence)
        inputs = self.tokenizer(ref_seq, return_tensors="pt", add_special_tokens=True).to(self.device)
        logits = self.model(**inputs).logits[0]

        all_logits: Dict[int, Dict[str, float]] = {}
        for pos_1b in range(1, len(ref_seq) + 1):
            pos_dict: Dict[str, float] = {}
            for aa in sorted(STANDARD_AMINO_ACIDS):
                tok_id = self.tokenizer.convert_tokens_to_ids(aa)
                pos_dict[aa] = logits[pos_1b, tok_id].item()
            all_logits[pos_1b] = pos_dict
        return all_logits


class MockProteinLanguageModel(ProteinLanguageModel):
    """Deterministic mock PLM for fast unit testing and offline environments."""

    def __init__(self, hidden_dim: int = 128) -> None:
        self.hidden_dim = hidden_dim

    def embed(
        self,
        sequences: List[str],
        batch_size: int = 16,
    ) -> SequenceEmbeddingResult:
        b = len(sequences)
        pooled = torch.zeros(b, self.hidden_dim)
        res_list = [torch.zeros(len(s), self.hidden_dim) for s in sequences]
        return SequenceEmbeddingResult(
            sequence_embeddings=pooled,
            residue_embeddings=res_list,
            hidden_dim=self.hidden_dim,
            model_name="mock_plm",
        )

    def score_mutations(
        self,
        sequence: str,
        mutations: List[Mutation],
        method: Literal["masked_marginal", "wildtype_marginal"] = "masked_marginal",
    ) -> List[MutationScoreResult]:
        ref_seq = validate_amino_acid_string(sequence)
        seq_len = len(ref_seq)
        results = []
        for mut in mutations:
            if mut.position_1based < 1 or mut.position_1based > seq_len:
                raise CoordinateOutOfBoundsError(mut.position_1based, seq_len, mut.to_string())
            actual_wt = ref_seq[mut.position_0based]
            if actual_wt != mut.ref_residue:
                raise SequenceMismatchError(mut.position_1based, mut.ref_residue, actual_wt, mut.to_string())
            results.append(
                MutationScoreResult(
                    mutation=mut,
                    log_odds_score=-1.5,
                    wt_log_prob=-0.5,
                    mut_log_prob=-2.0,
                    scoring_method=method,
                )
            )
        return results

    def score_combinatorial_variant(
        self,
        sequence: str,
        mutations: List[Mutation],
        method: Literal["additive", "joint_masked"] = "additive",
    ) -> CombinatorialScoreResult:
        single_scores = self.score_mutations(sequence, mutations)
        total = sum(s.log_odds_score for s in single_scores)
        return CombinatorialScoreResult(
            mutations=mutations,
            variant_sequence=sequence,
            score=total,
            additive_score=total,
            scoring_method=method,
        )

    def score_sequence_plausibility(self, sequence: str) -> float:
        return 4.2

    def get_masked_marginal_logits(
        self,
        sequence: str,
        position_1based: int,
    ) -> Dict[str, float]:
        ref_seq = validate_amino_acid_string(sequence)
        seq_len = len(ref_seq)
        if position_1based < 1 or position_1based > seq_len:
            raise CoordinateOutOfBoundsError(position_1based, seq_len, f"pos={position_1based}")
        wt = ref_seq[position_1based - 1]
        logits: Dict[str, float] = {}
        for aa in sorted(STANDARD_AMINO_ACIDS):
            if aa == wt:
                logits[aa] = 3.0
            else:
                # Deterministic pseudo-logit
                logits[aa] = float(((hash((position_1based, aa)) % 100) / 20.0) - 2.5)
        return logits

    def get_wildtype_marginal_logits(
        self,
        sequence: str,
    ) -> Dict[int, Dict[str, float]]:
        ref_seq = validate_amino_acid_string(sequence)
        all_logits: Dict[int, Dict[str, float]] = {}
        for pos_1b in range(1, len(ref_seq) + 1):
            all_logits[pos_1b] = self.get_masked_marginal_logits(ref_seq, pos_1b)
        return all_logits

