"""End-to-end integration test for Phase 4 closed-loop evolutionary pipeline with NSGA-II."""

from pathlib import Path
import pytest
import numpy as np

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.evolution.algorithms.nsga2 import NSGA2Optimizer
from evanthea.evolution.objectives import MultiObjectiveEvaluator, Objective, ObjectiveDirection
from evanthea.evolution.visualization import (
    extract_plot_data,
    render_generation_trajectory_table,
    render_pareto_summary_table,
)
from evanthea.mutation.constraints import ConstraintPipeline
from evanthea.mutation.point_mutations import ESM2PointMutationGenerator
from evanthea.mutation.proposals import CandidateProposalAPI
from evanthea.protein.language_model import MockProteinLanguageModel


def test_closed_loop_evolutionary_pipeline(tmp_path: Path):
    # Standard GFP reference snippet (length 80) including chromophore at 65-67
    wt = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ"
    mock_plm = MockProteinLanguageModel()

    # 1. Biological Constraints (GFP chromophore 65, 66, 67 protected, Cys disallowed)
    pipeline = ConstraintPipeline.create_gfp_pipeline(
        protected_positions={65, 66, 67},
        disallow_cysteine=True,
        max_charge_delta=2.0,
    )

    # 2. ESM-guided proposal generator
    gen = ESM2PointMutationGenerator(
        language_model=mock_plm,
        temperature=1.0,
        position_strategy="uniform",
        seed=42,
    )
    proposal_api = CandidateProposalAPI(
        operator=gen,
        protein_id="GFP_AEQVI",
        constraint_pipeline=pipeline,
    )

    # 3. Multi-objective evaluator (Mock surrogate fitness + ESM plausibility + Parsimony)
    class MockSurrogate:
        def predict_variants(self, variants):
            # Deterministic surrogate score favoring variants with mutations at positions 1-10
            scores = []
            for v in variants:
                s = 2.0
                for m in v.mutations:
                    if m.position_1based <= 10:
                        s += 0.5
                    else:
                        s -= 0.1
                scores.append(s)
            return scores

    evaluator = MultiObjectiveEvaluator.create_gfp_triad_evaluator(
        surrogate_predictor=MockSurrogate(),
        language_model=mock_plm,
        wildtype_sequence=wt,
    )

    # 4. Initialize and run NSGA-II
    optimizer = NSGA2Optimizer(
        evaluator=evaluator,
        proposal_api=proposal_api,
        wildtype_sequence=wt,
        population_size=10,
        max_generations=3,
        reference_point=[0.0, -10.0, -10.0],
        seed=42,
    )

    final_pop, history = optimizer.run()

    # 5. Assertions on population and history
    assert len(final_pop) == 10
    assert len(history.generations) == 4
    assert history.total_evaluations > 0

    # 6. Verify biological constraints were strictly respected across all evolved individuals
    for ind in final_pop:
        for m in ind.mutations:
            assert m.position_1based not in {65, 66, 67}, f"Chromophore violated: {m}"
            assert m.mut_residue != "C", f"Cysteine violation: {m}"

    # 7. Verify Pareto front
    pareto_front = final_pop.get_pareto_front()
    assert len(pareto_front) >= 1
    for ind in pareto_front:
        assert ind.rank == 0

    # 8. Test lineage and history serialization
    history_json = tmp_path / "history.json"
    history.save_json(history_json)
    assert history_json.exists()

    lineage_json = tmp_path / "lineage.json"
    optimizer.lineage_tracker.save_json(lineage_json)
    assert lineage_json.exists()

    # 9. Test visualization renderers
    traj_table = render_generation_trajectory_table(history, "predicted_fitness")
    assert "Generation" in traj_table
    assert "Gen 0" in traj_table

    pareto_table = render_pareto_summary_table(pareto_front)
    assert "Variant ID" in pareto_table

    plot_data = extract_plot_data(history)
    assert len(plot_data["generations"]) == 4
    assert len(plot_data["pairwise_diversity"]) == 4
