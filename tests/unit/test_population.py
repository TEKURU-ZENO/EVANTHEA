"""Unit tests for Individual and Population entities and Pareto dominance logic."""

import pytest
import numpy as np

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.evolution.population import Population


def _make_variant(seq: str, var_id: str, mutations: list[Mutation] | None = None) -> VariantRecord:
    muts = mutations or []
    return VariantRecord(
        variant_id=var_id,
        protein_id="GFP",
        mutations=muts,
        mutated_sequence=seq,
        is_wildtype=len(muts) == 0,
    )


def test_individual_properties_and_attributes():
    mut = Mutation(ref_residue="M", position_1based=1, mut_residue="A")
    var = _make_variant("ASKGE", "var_1", [mut])
    ind = Individual(variant=var, generation=2, parent_ids=["wt"])

    assert ind.id == "var_1"
    assert ind.sequence == "ASKGE"
    assert ind.num_mutations == 1
    assert ind.mutation_string == "M1A"
    assert ind.generation == 2
    assert ind.parent_ids == ["wt"]
    assert ind.is_evaluated is False


def test_individual_pareto_dominance():
    var1 = _make_variant("AAAAA", "v1")
    var2 = _make_variant("AAAAB", "v2")
    var3 = _make_variant("AAABB", "v3")

    ind1 = Individual(variant=var1, objectives_vector=[2.0, 3.0])
    ind2 = Individual(variant=var2, objectives_vector=[1.5, 2.5])
    ind3 = Individual(variant=var3, objectives_vector=[2.5, 2.0])

    # ind1 dominates ind2 (2.0 > 1.5 and 3.0 > 2.5)
    assert ind1.dominates(ind2) is True
    assert ind2.dominates(ind1) is False

    # ind1 vs ind3: ind1 is worse in obj 0 (2.0 < 2.5) but better in obj 1 (3.0 > 2.0)
    assert ind1.dominates(ind3) is False
    assert ind3.dominates(ind1) is False

    # Identical vectors do not dominate each other
    ind_dup = Individual(variant=var1, objectives_vector=[2.0, 3.0])
    assert ind1.dominates(ind_dup) is False


def test_individual_dominance_unevaluated_raises_error():
    ind = Individual(variant=_make_variant("A", "v1"))
    ind2 = Individual(variant=_make_variant("B", "v2"), objectives_vector=[1.0])
    with pytest.raises(ValueError):
        ind.dominates(ind2)


def test_population_containers_and_diversity():
    v1 = _make_variant("MSKGE", "v1")
    v2 = _make_variant("ASKGE", "v2", [Mutation(ref_residue="M", position_1based=1, mut_residue="A")])
    v3 = _make_variant("ASKGE", "v3", [Mutation(ref_residue="M", position_1based=1, mut_residue="A")])  # duplicate seq

    ind1 = Individual(variant=v1, fitness_values={"fit": 1.0}, is_evaluated=True, rank=0)
    ind2 = Individual(variant=v2, fitness_values={"fit": 2.5}, is_evaluated=True, rank=1)
    ind3 = Individual(variant=v3, fitness_values={"fit": 2.5}, is_evaluated=True, rank=0)

    pop = Population(generation=1)
    pop.add(ind1)
    pop.extend([ind2, ind3])

    assert len(pop) == 3
    assert pop.size == 3
    assert pop[0].id == "v1"

    # Deduplication
    unique_seqs = pop.unique_sequences()
    assert len(unique_seqs) == 2
    assert "MSKGE" in unique_seqs
    assert "ASKGE" in unique_seqs

    unique_inds = pop.unique_individuals()
    assert len(unique_inds) == 2

    # Diversity (Hamming distance between MSKGE and ASKGE is 1)
    diversity = pop.compute_pairwise_hamming_diversity()
    assert diversity == 1.0

    # Best by objective
    best = pop.best_by_objective("fit", maximize=True)
    assert best.id in ["v2", "v3"]
    assert best.fitness_values["fit"] == 2.5

    worst = pop.best_by_objective("fit", maximize=False)
    assert worst.id == "v1"

    # Pareto front (rank == 0)
    front = pop.get_pareto_front()
    assert len(front) == 2
    assert {i.id for i in front} == {"v1", "v3"}

    # Summary statistics
    stats = pop.summary_statistics()
    assert stats["generation"] == 1
    assert stats["population_size"] == 3
    assert stats["pareto_front_size"] == 2
    assert stats["objectives"]["fit"]["max"] == 2.5
    assert stats["objectives"]["fit"]["min"] == 1.0
