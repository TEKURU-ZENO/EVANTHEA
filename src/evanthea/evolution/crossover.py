"""Sequence-level and mutation-level recombination / crossover operators."""

from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple
import numpy as np

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.protein.sequence import (
    apply_mutations,
    compute_sequence_hash,
    validate_amino_acid_string,
)


def _reconstruct_mutations_against_ref(
    sequence: str,
    reference_sequence: str,
) -> List[Mutation]:
    """Finds all point substitutions in sequence relative to reference."""
    mutations: List[Mutation] = []
    for i, (ref_aa, mut_aa) in enumerate(zip(reference_sequence, sequence)):
        if ref_aa != mut_aa:
            mutations.append(
                Mutation(ref_residue=ref_aa, position_1based=i + 1, mut_residue=mut_aa)
            )
    return mutations


class CrossoverOperator(ABC):
    """Abstract base class for genetic sequence recombination."""

    def __init__(self, reference_sequence: str, seed: Optional[int] = None) -> None:
        self.reference_sequence = validate_amino_acid_string(reference_sequence)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        generation: int = 0,
    ) -> Tuple[Individual, Individual]:
        """Recombines two parent individuals to produce two offspring individuals."""
        raise NotImplementedError


class SinglePointCrossover(CrossoverOperator):
    """Classical single-point crossover slicing two sequences at a random homologous coordinate."""

    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        generation: int = 0,
    ) -> Tuple[Individual, Individual]:
        seq1 = parent1.sequence
        seq2 = parent2.sequence
        l1, l2 = len(seq1), len(seq2)

        if l1 != l2:
            raise ValueError(f"SinglePointCrossover requires equal length parents (got {l1} and {l2})")

        if l1 <= 1:
            return parent1, parent2

        c = int(self.rng.integers(1, l1))

        child_seq1 = seq1[:c] + seq2[c:]
        child_seq2 = seq2[:c] + seq1[c:]

        # Reconstruct mutation lists against reference
        muts1 = _reconstruct_mutations_against_ref(child_seq1, self.reference_sequence)
        muts2 = _reconstruct_mutations_against_ref(child_seq2, self.reference_sequence)

        h1 = compute_sequence_hash(child_seq1)
        h2 = compute_sequence_hash(child_seq2)

        v1 = VariantRecord(
            variant_id=h1[:16],
            protein_id=parent1.variant.protein_id,
            mutations=muts1,
            mutated_sequence=child_seq1,
            is_wildtype=len(muts1) == 0,
            num_mutations=len(muts1),
            metadata={"operator": "single_point_crossover", "crossover_point": c},
        )
        v2 = VariantRecord(
            variant_id=h2[:16],
            protein_id=parent2.variant.protein_id,
            mutations=muts2,
            mutated_sequence=child_seq2,
            is_wildtype=len(muts2) == 0,
            num_mutations=len(muts2),
            metadata={"operator": "single_point_crossover", "crossover_point": c},
        )

        ind1 = Individual(
            variant=v1,
            generation=generation,
            parent_ids=[parent1.id, parent2.id],
        )
        ind2 = Individual(
            variant=v2,
            generation=generation,
            parent_ids=[parent1.id, parent2.id],
        )

        return ind1, ind2


class HomologousMutationCrossover(CrossoverOperator):
    """Protein-engineering crossover recombining sets of point mutations from both parents."""

    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        generation: int = 0,
    ) -> Tuple[Individual, Individual]:
        # Map positions to mutations
        m1_map = {m.position_1based: m for m in parent1.mutations}
        m2_map = {m.position_1based: m for m in parent2.mutations}

        all_positions = sorted(set(m1_map.keys()) | set(m2_map.keys()))

        child1_muts: List[Mutation] = []
        child2_muts: List[Mutation] = []

        for pos in all_positions:
            in_m1 = pos in m1_map
            in_m2 = pos in m2_map

            if in_m1 and in_m2:
                # Both parents have mutation at this position
                if m1_map[pos].mut_residue == m2_map[pos].mut_residue:
                    # Identical mutation: pass to both
                    child1_muts.append(m1_map[pos])
                    child2_muts.append(m2_map[pos])
                else:
                    # Conflicting substitutions: swap or flip
                    if self.rng.random() < 0.5:
                        child1_muts.append(m1_map[pos])
                        child2_muts.append(m2_map[pos])
                    else:
                        child1_muts.append(m2_map[pos])
                        child2_muts.append(m1_map[pos])
            elif in_m1:
                # Mutation present only in parent 1
                if self.rng.random() < 0.5:
                    child1_muts.append(m1_map[pos])
                else:
                    child2_muts.append(m1_map[pos])
            elif in_m2:
                # Mutation present only in parent 2
                if self.rng.random() < 0.5:
                    child1_muts.append(m2_map[pos])
                else:
                    child2_muts.append(m2_map[pos])

        # Sort and build sequences
        child1_muts.sort(key=lambda m: m.position_1based)
        child2_muts.sort(key=lambda m: m.position_1based)

        seq1 = apply_mutations(self.reference_sequence, child1_muts)
        seq2 = apply_mutations(self.reference_sequence, child2_muts)

        h1 = compute_sequence_hash(seq1)
        h2 = compute_sequence_hash(seq2)

        v1 = VariantRecord(
            variant_id=h1[:16],
            protein_id=parent1.variant.protein_id,
            mutations=child1_muts,
            mutated_sequence=seq1,
            is_wildtype=len(child1_muts) == 0,
            num_mutations=len(child1_muts),
            metadata={"operator": "homologous_mutation_crossover"},
        )
        v2 = VariantRecord(
            variant_id=h2[:16],
            protein_id=parent2.variant.protein_id,
            mutations=child2_muts,
            mutated_sequence=seq2,
            is_wildtype=len(child2_muts) == 0,
            num_mutations=len(child2_muts),
            metadata={"operator": "homologous_mutation_crossover"},
        )

        return (
            Individual(variant=v1, generation=generation, parent_ids=[parent1.id, parent2.id]),
            Individual(variant=v2, generation=generation, parent_ids=[parent1.id, parent2.id]),
        )
