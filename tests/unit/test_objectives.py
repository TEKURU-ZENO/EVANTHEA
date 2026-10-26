"""Unit tests for MultiObjectiveEvaluator and objective normalization."""

import pytest

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import (
    MultiObjectiveEvaluator,
    Objective,
    ObjectiveDirection,
)
from evanthea.protein.language_model import MockProteinLanguageModel


def _make_individual(seq: str, var_id: str, muts: list[Mutation] | None = None) -> Individual:
    v = VariantRecord(
        variant_id=var_id,
        protein_id="GFP",
        mutations=muts or [],
        mutated_sequence=seq,
        is_wildtype=not bool(muts),
    )
    return Individual(variant=v)


def test_objective_normalization():
    obj_max = Objective(name="score", direction=ObjectiveDirection.MAXIMIZE)
    assert obj_max.normalize_for_maximization(5.5) == 5.5

    obj_min = Objective(name="cost", direction=ObjectiveDirection.MINIMIZE)
    assert obj_min.normalize_for_maximization(3.0) == -3.0


def test_multi_objective_evaluator_basic():
    obj1 = Objective(name="f1", direction=ObjectiveDirection.MAXIMIZE)
    obj2 = Objective(name="f2", direction=ObjectiveDirection.MINIMIZE)

    evaluator = MultiObjectiveEvaluator(
        objectives=[obj1, obj2],
        evaluators={
            "f1": lambda ind: len(ind.sequence) * 1.5,
            "f2": lambda ind: ind.num_mutations * 2.0,
        },
    )

    ind = _make_individual("MSKGE", "v1", [Mutation(ref_residue="M", position_1based=1, mut_residue="A")])
    evaluator.evaluate_individual(ind)

    assert ind.is_evaluated is True
    assert ind.fitness_values["f1"] == 5 * 1.5
    assert ind.fitness_values["f2"] == 1 * 2.0
    # Normalized for maximization: [7.5, -2.0]
    assert ind.objectives_vector == [7.5, -2.0]


def test_multi_objective_evaluator_validation():
    with pytest.raises(ValueError):
        MultiObjectiveEvaluator(objectives=[], evaluators={})

    obj1 = Objective(name="dup")
    obj2 = Objective(name="dup")
    with pytest.raises(ValueError):
        MultiObjectiveEvaluator(objectives=[obj1, obj2], evaluators={"dup": lambda x: 1.0})

    obj3 = Objective(name="missing")
    with pytest.raises(ValueError):
        MultiObjectiveEvaluator(objectives=[obj3], evaluators={})


def test_gfp_triad_evaluator_factory():
    mock_plm = MockProteinLanguageModel()

    class MockSurrogate:
        def predict_variants(self, variants):
            return [3.14 for _ in variants]

    wt = "MSKGE"
    evaluator = MultiObjectiveEvaluator.create_gfp_triad_evaluator(
        surrogate_predictor=MockSurrogate(),
        language_model=mock_plm,
        wildtype_sequence=wt,
    )

    mut = Mutation(ref_residue="M", position_1based=1, mut_residue="A")
    ind = _make_individual("ASKGE", "v_mut", [mut])

    evaluator.evaluate_individual(ind)
    assert ind.is_evaluated is True
    assert ind.fitness_values["predicted_fitness"] == pytest.approx(3.14)
    assert ind.fitness_values["mutational_parsimony"] == -1.0
    assert "esm2_plausibility" in ind.fitness_values
    assert len(ind.objectives_vector) == 3
