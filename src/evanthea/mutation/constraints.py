"""Biological and biophysical constraints for candidate mutation proposals."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from evanthea.constants import STANDARD_AMINO_ACIDS
from evanthea.data.schemas import Mutation
from evanthea.mutation.base import ProposalRecord

# Residue charge mapping at physiological pH (~7.4)
PHYSIOLOGICAL_CHARGE: Dict[str, float] = {
    "R": 1.0,
    "K": 1.0,
    "H": 0.1,
    "D": -1.0,
    "E": -1.0,
}


def compute_net_charge(sequence: str) -> float:
    """Computes total net electrostatic charge of an amino acid sequence at pH ~7.4."""
    return sum(PHYSIOLOGICAL_CHARGE.get(aa, 0.0) for aa in sequence)


def compute_charge_delta(mutations: List[Mutation]) -> float:
    """Computes net charge change resulting from a list of point mutations."""
    delta = 0.0
    for m in mutations:
        wt_charge = PHYSIOLOGICAL_CHARGE.get(m.ref_residue, 0.0)
        mut_charge = PHYSIOLOGICAL_CHARGE.get(m.mut_residue, 0.0)
        delta += (mut_charge - wt_charge)
    return delta


class BaseConstraint(ABC):
    """Abstract base class for all hard biological constraints and soft biophysical filters."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def evaluate(
        self,
        parent_sequence: str,
        mutations: List[Mutation],
        candidate_sequence: str,
        proposal: Optional[ProposalRecord] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Evaluates whether candidate satisfies this biological constraint.

        Args:
            parent_sequence: Full wild-type / parent sequence.
            mutations: List of applied point mutations.
            candidate_sequence: Full reconstructed child sequence.
            proposal: Optional ProposalRecord containing generator provenance and scores.

        Returns:
            Tuple of (is_valid: bool, rejection_reason: Optional[str]).
        """
        raise NotImplementedError


class ProtectedPositionsConstraint(BaseConstraint):
    """Rejects mutations targeting biologically critical coordinates (e.g. active site or chromophore)."""

    def __init__(self, protected_positions: Set[int]) -> None:
        super().__init__(
            name="protected_positions",
            description=f"Protects coordinates: {sorted(protected_positions)}",
        )
        self.protected_positions = set(protected_positions)

    def evaluate(
        self,
        parent_sequence: str,
        mutations: List[Mutation],
        candidate_sequence: str,
        proposal: Optional[ProposalRecord] = None,
    ) -> Tuple[bool, Optional[str]]:
        violating = [m for m in mutations if m.position_1based in self.protected_positions]
        if violating:
            pos_strs = [f"{m.to_string()} (pos {m.position_1based})" for m in violating]
            return False, f"Violated protected positions {sorted(self.protected_positions)}: {', '.join(pos_strs)}"
        return True, None


class DisallowedResiduesConstraint(BaseConstraint):
    """Rejects mutations introducing disallowed amino acids (e.g. unpaired Cysteine to avoid aggregation)."""

    def __init__(self, disallowed_residues: Set[str]) -> None:
        super().__init__(
            name="disallowed_residues",
            description=f"Disallows introducing residues: {sorted(disallowed_residues)}",
        )
        self.disallowed_residues = set(r.upper() for r in disallowed_residues)

    def evaluate(
        self,
        parent_sequence: str,
        mutations: List[Mutation],
        candidate_sequence: str,
        proposal: Optional[ProposalRecord] = None,
    ) -> Tuple[bool, Optional[str]]:
        violating = [m for m in mutations if m.mut_residue in self.disallowed_residues]
        if violating:
            pos_strs = [f"{m.to_string()}" for m in violating]
            return False, f"Disallowed amino acid(s) introduced ({sorted(self.disallowed_residues)}): {', '.join(pos_strs)}"
        return True, None


class AllowedResiduesPerPositionConstraint(BaseConstraint):
    """Restricts mutations at specified positions to a strictly allowed vocabulary."""

    def __init__(self, allowed_map: Dict[int, Set[str]]) -> None:
        super().__init__(
            name="allowed_residues_per_position",
            description="Restricts position-specific substitution vocabulary.",
        )
        self.allowed_map = {pos: set(aa.upper() for aa in aas) for pos, aas in allowed_map.items()}

    def evaluate(
        self,
        parent_sequence: str,
        mutations: List[Mutation],
        candidate_sequence: str,
        proposal: Optional[ProposalRecord] = None,
    ) -> Tuple[bool, Optional[str]]:
        for m in mutations:
            if m.position_1based in self.allowed_map:
                allowed_set = self.allowed_map[m.position_1based]
                if m.mut_residue not in allowed_set:
                    return False, f"Residue '{m.mut_residue}' at position {m.position_1based} not in allowed set {sorted(allowed_set)}"
        return True, None


class MaxMutationCountConstraint(BaseConstraint):
    """Bounds the maximum Hamming distance / substitution count from the parent sequence."""

    def __init__(self, max_mutations: int) -> None:
        super().__init__(
            name="max_mutation_count",
            description=f"Caps maximum mutations at {max_mutations}",
        )
        if max_mutations < 1:
            raise ValueError(f"max_mutations must be >= 1, got {max_mutations}")
        self.max_mutations = max_mutations

    def evaluate(
        self,
        parent_sequence: str,
        mutations: List[Mutation],
        candidate_sequence: str,
        proposal: Optional[ProposalRecord] = None,
    ) -> Tuple[bool, Optional[str]]:
        if len(mutations) > self.max_mutations:
            return False, f"Mutation count ({len(mutations)}) exceeds maximum allowed ({self.max_mutations})"
        return True, None


class ChargeDeltaConstraint(BaseConstraint):
    """Prevents severe electrostatic disruption by bounding net charge deviation at pH 7.4."""

    def __init__(self, max_abs_delta: float = 2.0) -> None:
        super().__init__(
            name="charge_delta",
            description=f"Limits net charge shift to |Δcharge| <= {max_abs_delta:.2f}",
        )
        if max_abs_delta < 0:
            raise ValueError(f"max_abs_delta must be non-negative, got {max_abs_delta}")
        self.max_abs_delta = max_abs_delta

    def evaluate(
        self,
        parent_sequence: str,
        mutations: List[Mutation],
        candidate_sequence: str,
        proposal: Optional[ProposalRecord] = None,
    ) -> Tuple[bool, Optional[str]]:
        delta = compute_charge_delta(mutations)
        if abs(delta) > self.max_abs_delta + 1e-6:
            return False, f"Net charge shift |{delta:.2f}| exceeds maximum allowed |{self.max_abs_delta:.2f}|"
        return True, None


class MinimumLikelihoodConstraint(BaseConstraint):
    """Filters out candidate proposals whose PLM marginal log-odds fall below a viability threshold."""

    def __init__(self, min_delta_log_p: float = -4.0) -> None:
        super().__init__(
            name="minimum_likelihood",
            description=f"Requires PLM Δ log p >= {min_delta_log_p:.2f}",
        )
        self.min_delta_log_p = min_delta_log_p

    def evaluate(
        self,
        parent_sequence: str,
        mutations: List[Mutation],
        candidate_sequence: str,
        proposal: Optional[ProposalRecord] = None,
    ) -> Tuple[bool, Optional[str]]:
        if proposal is None:
            return True, None

        # Check step scores
        if proposal.step_proposal_scores:
            lowest_step = min(proposal.step_proposal_scores)
            if lowest_step < self.min_delta_log_p:
                return False, f"Minimum step score ({lowest_step:.3f}) falls below cutoff ({self.min_delta_log_p:.3f})"

        # Check final variant score
        if proposal.final_variant_score is not None:
            if proposal.final_variant_score < self.min_delta_log_p:
                return False, f"Final variant score ({proposal.final_variant_score:.3f}) falls below cutoff ({self.min_delta_log_p:.3f})"

        return True, None


class ConstraintPipeline:
    """Chains multiple biological and biophysical constraints, auditing all violation causes."""

    def __init__(
        self,
        constraints: Optional[List[BaseConstraint]] = None,
        early_stop: bool = False,
    ) -> None:
        self.constraints: List[BaseConstraint] = list(constraints) if constraints else []
        self.early_stop = early_stop

    def add_constraint(self, constraint: BaseConstraint) -> None:
        """Adds a constraint to the end of the evaluation chain."""
        self.constraints.append(constraint)

    def evaluate(
        self,
        parent_sequence: str,
        mutations: List[Mutation],
        candidate_sequence: str,
        proposal: Optional[ProposalRecord] = None,
    ) -> Tuple[bool, List[str]]:
        """Evaluates all configured constraints against candidate variant."""
        rejection_reasons: List[str] = []

        for c in self.constraints:
            valid, reason = c.evaluate(parent_sequence, mutations, candidate_sequence, proposal)
            if not valid and reason:
                rejection_reasons.append(reason)
                if self.early_stop:
                    break

        is_accepted = len(rejection_reasons) == 0
        return is_accepted, rejection_reasons

    def filter_proposals(
        self,
        proposals: List[ProposalRecord],
        parent_sequence: str,
    ) -> Tuple[List[ProposalRecord], List[ProposalRecord]]:
        """Evaluates and splits a batch of proposals into accepted and rejected subsets."""
        accepted: List[ProposalRecord] = []
        rejected: List[ProposalRecord] = []

        for prop in proposals:
            is_valid, reasons = self.evaluate(
                parent_sequence=parent_sequence,
                mutations=prop.mutations,
                candidate_sequence=prop.child_sequence,
                proposal=prop,
            )
            prop.is_accepted = is_valid
            prop.rejection_reasons = reasons

            if is_valid:
                accepted.append(prop)
            else:
                rejected.append(prop)

        return accepted, rejected

    @classmethod
    def create_gfp_pipeline(
        cls,
        protected_positions: Optional[Set[int]] = None,
        disallow_cysteine: bool = False,
        max_charge_delta: float = 2.0,
        max_mutations: int = 5,
        min_delta_log_p: Optional[float] = None,
    ) -> "ConstraintPipeline":
        """Factory function configuring biologically validated defaults for GFP evolution."""
        # Standard GFP fluorophore triad at residues 65-67 (Thr65-Tyr66-Gly67 in avGFP)
        prot_pos = set(protected_positions) if protected_positions is not None else {65, 66, 67}

        constraints: List[BaseConstraint] = [
            ProtectedPositionsConstraint(protected_positions=prot_pos),
            MaxMutationCountConstraint(max_mutations=max_mutations),
            ChargeDeltaConstraint(max_abs_delta=max_charge_delta),
        ]

        if disallow_cysteine:
            constraints.append(DisallowedResiduesConstraint(disallowed_residues={"C"}))

        if min_delta_log_p is not None:
            constraints.append(MinimumLikelihoodConstraint(min_delta_log_p=min_delta_log_p))

        return cls(constraints=constraints)
