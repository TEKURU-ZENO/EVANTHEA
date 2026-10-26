"""Multi-objective fitness evaluation subsystem for protein sequence optimization."""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from evanthea.data.schemas import VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.protein.language_model import ProteinLanguageModel
from evanthea.protein.sequence import compute_hamming_distance


class ObjectiveDirection(str, Enum):
    """Optimization direction for an evolutionary objective."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class Objective(BaseModel):
    """Specification of an optimization objective."""
    name: str = Field(..., description="Unique objective identifier")
    direction: ObjectiveDirection = Field(
        ObjectiveDirection.MAXIMIZE,
        description="Whether to maximize or minimize this objective"
    )
    weight: float = Field(1.0, description="Optional weighting factor for scalarized aggregation")
    description: str = Field("", description="Scientific description of the target property")

    def normalize_for_maximization(self, value: float) -> float:
        """Converts raw objective score into a maximization scale (negates if MINIMIZE)."""
        return float(value) if self.direction == ObjectiveDirection.MAXIMIZE else -float(value)


class MultiObjectiveEvaluator:
    """Orchestrates multi-objective fitness evaluation across learned surrogates, PLMs, and sequence properties."""

    def __init__(
        self,
        objectives: List[Objective],
        evaluators: Dict[str, Callable[[Individual], float]],
    ) -> None:
        if not objectives:
            raise ValueError("MultiObjectiveEvaluator requires at least one Objective.")

        obj_names = [o.name for o in objectives]
        if len(obj_names) != len(set(obj_names)):
            raise ValueError(f"Duplicate objective names detected: {obj_names}")

        for o in objectives:
            if o.name not in evaluators:
                raise ValueError(f"Missing evaluation callable for objective '{o.name}'")

        self.objectives = list(objectives)
        self.evaluators = dict(evaluators)

    def evaluate_individual(self, individual: Individual) -> Individual:
        """Evaluates all registered objectives for a single individual."""
        raw_values: Dict[str, float] = {}
        norm_vector: List[float] = []

        for obj in self.objectives:
            eval_fn = self.evaluators[obj.name]
            raw_score = float(eval_fn(individual))
            raw_values[obj.name] = raw_score
            norm_vector.append(obj.normalize_for_maximization(raw_score))

        individual.fitness_values = raw_values
        individual.objectives_vector = norm_vector
        individual.is_evaluated = True
        return individual

    def evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """Evaluates all individuals in a population that have not yet been evaluated."""
        for ind in population:
            if not ind.is_evaluated:
                self.evaluate_individual(ind)
        return population

    @classmethod
    def create_gfp_triad_evaluator(
        cls,
        surrogate_predictor: Any,
        language_model: ProteinLanguageModel,
        wildtype_sequence: str,
        feature_builder: Optional[Any] = None,
    ) -> "MultiObjectiveEvaluator":
        """Factory creating the canonical 3-objective EVANTHEA evaluator on GFP:
        1. predicted_fitness (Surrogate Assay Model) [MAXIMIZE]
        2. esm2_plausibility (ESM-2 Masked Marginal Log-Odds) [MAXIMIZE]
        3. mutational_parsimony (-Hamming distance) [MAXIMIZE]
        """
        objectives = [
            Objective(
                name="predicted_fitness",
                direction=ObjectiveDirection.MAXIMIZE,
                description="Learned surrogate model prediction of assay fitness (fluorescence)",
            ),
            Objective(
                name="esm2_plausibility",
                direction=ObjectiveDirection.MAXIMIZE,
                description="Zero-shot masked-marginal log-odds plausibility from ESM-2",
            ),
            Objective(
                name="mutational_parsimony",
                direction=ObjectiveDirection.MAXIMIZE,
                description="Mutational parsimony relative to wild-type (-Hamming distance)",
            ),
        ]

        def _eval_surrogate(ind: Individual) -> float:
            if hasattr(surrogate_predictor, "predict_variants"):
                preds = surrogate_predictor.predict_variants([ind.variant])
                return float(preds[0])
            elif hasattr(surrogate_predictor, "predict"):
                if feature_builder is not None:
                    if hasattr(feature_builder, "build_features_for_variant"):
                        feat_vec = feature_builder.build_features_for_variant(ind.variant, "mutation_features")
                        feat_matrix = np.asarray(feat_vec, dtype=np.float32).reshape(1, -1)
                    elif hasattr(feature_builder, "build_matrix"):
                        feat_matrix, _ = feature_builder.build_matrix([ind.variant], "mutation_features")
                    else:
                        feat_matrix, _ = feature_builder.build_feature_matrix([ind.variant])
                    preds = surrogate_predictor.predict(feat_matrix)
                    return float(preds[0])
                else:
                    preds = surrogate_predictor.predict([ind.sequence])
                    return float(preds[0])
            return 0.0

        def _eval_esm2(ind: Individual) -> float:
            # If variant was proposed by ESM2PointMutationGenerator with scores recorded
            prop_meta = ind.variant.metadata.get("proposal", {})
            if "final_variant_score" in prop_meta and prop_meta["final_variant_score"] is not None:
                return float(prop_meta["final_variant_score"])

            if not ind.mutations:
                return 0.0

            try:
                scores = language_model.score_mutations(
                    wildtype_sequence,
                    ind.mutations,
                    method="masked_marginal",
                )
                return float(sum(s.log_odds_score for s in scores))
            except Exception:
                return -10.0

        def _eval_parsimony(ind: Individual) -> float:
            # Negative Hamming distance so that fewer mutations = higher parsimony
            dist = compute_hamming_distance(ind.sequence, wildtype_sequence)
            return float(-dist)

        evaluators = {
            "predicted_fitness": _eval_surrogate,
            "esm2_plausibility": _eval_esm2,
            "mutational_parsimony": _eval_parsimony,
        }

        return cls(objectives=objectives, evaluators=evaluators)
