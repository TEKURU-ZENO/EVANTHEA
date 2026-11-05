"""Gymnasium-compliant protein variant design environment with action masking and evolutionary memory integration."""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from evanthea.constants import STANDARD_AMINO_ACIDS
from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator
from evanthea.memory.manager import EvolutionaryMemory
from evanthea.mutation.constraints import ConstraintPipeline
from evanthea.protein.sequence import validate_amino_acid_string
from evanthea.rl.rewards import CompositeRewardFunction, RewardWeights
from evanthea.rl.schema import ActionDefinition, RLStepResult


# Compatibility layer for standard RL spaces
class DiscreteSpace:
    """Lightweight discrete space compatible with gymnasium.spaces.Discrete."""
    def __init__(self, n: int, seed: Optional[int] = None) -> None:
        self.n = n
        self.dtype = np.int64
        self.shape = ()
        self.rng = np.random.default_rng(seed)

    def sample(self, mask: Optional[np.ndarray] = None) -> int:
        if mask is not None:
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                return 0
            return int(self.rng.choice(valid_indices))
        return int(self.rng.integers(0, self.n))

    def contains(self, x: Any) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n


class BoxSpace:
    """Lightweight box space compatible with gymnasium.spaces.Box."""
    def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: Any = np.float32) -> None:
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    def contains(self, x: Any) -> bool:
        arr = np.asarray(x)
        return arr.shape == self.shape


class ProteinDesignEnv:
    """Gymnasium-compliant Markov Decision Process for directed protein variant generation.

    State: Vectorized representation of candidate sequence, mutational accumulation, and fitness state.
    Action: Discrete selection of (sequence position, replacement amino acid).
    Reward: Multi-objective composite balancing fitness, PLM plausibility, novelty, and parsimony.
    Masking: Dynamic action mask eliminating wild-type self-identity loops and illegal substitutions.
    """

    def __init__(
        self,
        wildtype_sequence: str,
        evaluator: MultiObjectiveEvaluator,
        reward_weights: Optional[RewardWeights] = None,
        memory: Optional[EvolutionaryMemory] = None,
        constraint_pipeline: Optional[ConstraintPipeline] = None,
        allowed_positions: Optional[Sequence[int]] = None,
        max_mutations: int = 5,
        allow_revisits: bool = False,
        allow_stop_action: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.wildtype_sequence = validate_amino_acid_string(wildtype_sequence)
        self.seq_len = len(self.wildtype_sequence)
        self.evaluator = evaluator
        self.memory = memory
        self.constraint_pipeline = constraint_pipeline
        self.max_mutations = max_mutations
        self.allow_revisits = allow_revisits
        self.allow_stop_action = allow_stop_action
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

        # 1. Determine mutable position indices (0-based)
        if allowed_positions is not None:
            self.mutable_positions = sorted(list(set(allowed_positions)))
            for p in self.mutable_positions:
                if not (0 <= p < self.seq_len):
                    raise ValueError(f"Allowed position {p} out of bounds for sequence length {self.seq_len}")
        else:
            self.mutable_positions = list(range(self.seq_len))

        # Canonical amino acids sorted for deterministic indexing
        self.amino_acids = sorted(list(STANDARD_AMINO_ACIDS))
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}

        # 2. Build Action Map: action_id -> ActionDefinition
        self.action_to_def: Dict[int, ActionDefinition] = {}
        self.def_to_action: Dict[Tuple[int, str], int] = {}
        action_idx = 0

        for pos in self.mutable_positions:
            for aa in self.amino_acids:
                act_def = ActionDefinition(
                    action_id=action_idx,
                    position_1based=pos + 1,
                    position_0based=pos,
                    ref_residue=self.wildtype_sequence[pos],
                    mut_residue=aa,
                )
                self.action_to_def[action_idx] = act_def
                self.def_to_action[(pos, aa)] = action_idx
                action_idx += 1

        self.stop_action_id: Optional[int] = None
        if self.allow_stop_action:
            self.stop_action_id = action_idx
            action_idx += 1

        self.num_actions = action_idx
        self.action_space = DiscreteSpace(self.num_actions, seed=seed)

        # 3. Observation Space definition
        # Observation layout:
        # - Flat sequence encoding of mutable positions: len(mutable_positions) * 20
        # - Mutation count scalar: 1
        # - Remaining horizon scalar: 1
        # - Current predicted fitness: 1
        # - Current ESM-2 score: 1
        self.seq_feature_dim = len(self.mutable_positions) * len(self.amino_acids)
        self.scalar_feature_dim = 4
        self.obs_dim = self.seq_feature_dim + self.scalar_feature_dim
        self.observation_space = BoxSpace(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # 4. Initialize reward function
        self.reward_fn = CompositeRewardFunction(
            evaluator=self.evaluator,
            weights=reward_weights,
            memory=self.memory,
        )

        # Episode state variables
        self.current_sequence: str = self.wildtype_sequence
        self.accumulated_mutations: List[Mutation] = []
        self.modified_positions: Set[int] = set()
        self.current_fitness: float = 0.0
        self.current_esm_score: float = 0.0
        self.step_count: int = 0
        self.evaluations_count: int = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets environment to starting wild-type sequence and returns initial observation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.rng = np.random.default_rng(seed)

        self.current_sequence = self.wildtype_sequence
        self.accumulated_mutations = []
        self.modified_positions = set()
        self.step_count = 0

        # Evaluate initial wild-type state (bypassing if cached in memory)
        cached_rec = self.memory.lookup_exact(self.wildtype_sequence) if self.memory is not None else None
        if cached_rec is not None and cached_rec.fitness_values:
            self.current_fitness = float(cached_rec.fitness_values.get("predicted_fitness", 0.0))
            self.current_esm_score = float(cached_rec.fitness_values.get("esm2_plausibility", cached_rec.fitness_values.get("esm2_log_plausibility", 0.0)))
        else:
            wt_ind = Individual(
                variant=VariantRecord(
                    variant_id="WT_INIT",
                    protein_id="DESIGN",
                    mutations=[],
                    mutated_sequence=self.wildtype_sequence,
                    is_wildtype=True,
                )
            )
            self.evaluator.evaluate_individual(wt_ind)
            self.evaluations_count += 1
            self.current_fitness = float(wt_ind.fitness_values.get("predicted_fitness", 0.0))
            self.current_esm_score = float(wt_ind.fitness_values.get("esm2_plausibility", wt_ind.fitness_values.get("esm2_log_plausibility", 0.0)))
            if self.memory is not None:
                self.memory.register_candidate(wt_ind)

        obs = self._get_observation()
        info = {
            "sequence": self.current_sequence,
            "mutations": [],
            "current_fitness": self.current_fitness,
            "current_esm_score": self.current_esm_score,
            "action_mask": self.get_action_mask(),
        }
        return obs, info

    def get_action_mask(self) -> np.ndarray:
        """Returns boolean mask where True indicates a valid legal action."""
        mask = np.ones(self.num_actions, dtype=bool)

        for act_id, act_def in self.action_to_def.items():
            pos = act_def.position_0based
            mut_aa = act_def.mut_residue

            # 1. Mask wild-type identity (cannot mutate residue to itself)
            if self.current_sequence[pos] == mut_aa:
                mask[act_id] = False
                continue

            # 2. Mask already modified positions if revisits disallowed
            if not self.allow_revisits and pos in self.modified_positions:
                mask[act_id] = False
                continue

            # 3. Check constraint pipeline if active
            if self.constraint_pipeline is not None:
                # Test candidate mutation
                test_mut = Mutation(
                    ref_residue=self.wildtype_sequence[pos],
                    position_1based=pos + 1,
                    mut_residue=mut_aa,
                )
                test_seq = self.current_sequence[:pos] + mut_aa + self.current_sequence[pos + 1:]
                test_muts = [m for m in self.accumulated_mutations if m.position_0based != pos]
                if mut_aa != self.wildtype_sequence[pos]:
                    test_muts.append(test_mut)

                is_valid, _ = self.constraint_pipeline.evaluate(
                    parent_sequence=self.wildtype_sequence,
                    mutations=test_muts,
                    candidate_sequence=test_seq,
                )
                if not is_valid:
                    mask[act_id] = False

        # STOP action is always valid if allowed
        if self.stop_action_id is not None:
            mask[self.stop_action_id] = True

        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes a single mutational transition step in the protein design MDP."""
        if not (0 <= action < self.num_actions):
            raise ValueError(f"Action {action} is out of bounds for action space of size {self.num_actions}")

        # Check action mask
        mask = self.get_action_mask()
        if not mask[action]:
            # Illegal action penalty
            obs = self._get_observation()
            return obs, -5.0, False, False, {"error": "illegal_action_taken", "action": action}

        # Handle STOP action
        if self.stop_action_id is not None and action == self.stop_action_id:
            obs = self._get_observation()
            info = {
                "sequence": self.current_sequence,
                "mutations": [m.model_dump() for m in self.accumulated_mutations],
                "fitness": self.current_fitness,
                "esm_score": self.current_esm_score,
                "stopped_early": True,
            }
            return obs, 0.0, True, False, info

        # Apply mutation
        act_def = self.action_to_def[action]
        pos = act_def.position_0based
        mut_aa = act_def.mut_residue
        ref_aa = self.current_sequence[pos]

        next_sequence = self.current_sequence[:pos] + mut_aa + self.current_sequence[pos + 1:]
        new_mutation = Mutation(
            ref_residue=self.wildtype_sequence[pos],
            position_1based=pos + 1,
            mut_residue=mut_aa,
        )

        # Update accumulated mutations
        next_mutations = [m for m in self.accumulated_mutations if m.position_0based != pos]
        if mut_aa != self.wildtype_sequence[pos]:
            next_mutations.append(new_mutation)

        # Validate constraints
        is_feasible = True
        violations: List[str] = []
        if self.constraint_pipeline is not None:
            is_feasible, violations = self.constraint_pipeline.evaluate(
                parent_sequence=self.wildtype_sequence,
                mutations=next_mutations,
                candidate_sequence=next_sequence,
            )

        # Compute multi-objective reward with memory bypass
        cached_rec = self.memory.lookup_exact(next_sequence) if self.memory is not None else None
        if cached_rec is not None and cached_rec.fitness_values:
            curr_fitness = float(cached_rec.fitness_values.get("predicted_fitness", 0.0))
            esm_score = float(cached_rec.fitness_values.get("esm2_plausibility", cached_rec.fitness_values.get("esm2_log_plausibility", 0.0)))
            delta_fitness = curr_fitness - self.current_fitness
            fitness_reward = self.reward_fn.weights.fitness_weight * delta_fitness
            esm_reward = self.reward_fn.weights.esm_plausibility_weight * esm_score
            novelty_val = self.memory.get_novelty_score(next_sequence) if self.reward_fn.weights.novelty_weight > 0.0 else 0.0
            novelty_reward = self.reward_fn.weights.novelty_weight * novelty_val
            cons_penalty = self.reward_fn.weights.conservation_penalty
            viol_penalty = self.reward_fn.weights.violation_penalty * len(violations) if not is_feasible else 0.0
            raw_total = fitness_reward + esm_reward + novelty_reward - cons_penalty - viol_penalty
            if self.reward_fn.clip_reward is not None:
                reward = float(np.clip(raw_total, self.reward_fn.clip_reward[0], self.reward_fn.clip_reward[1]))
            else:
                reward = float(raw_total)
            breakdown = {
                "total_reward": reward,
                "delta_fitness": delta_fitness,
                "current_fitness": curr_fitness,
                "esm_score": esm_score,
                "novelty_score": novelty_val,
                "fitness_reward": fitness_reward,
                "esm_reward": esm_reward,
                "novelty_reward": novelty_reward,
                "conservation_penalty": cons_penalty,
                "violation_penalty": viol_penalty,
            }
        else:
            reward, breakdown = self.reward_fn.compute_reward(
                current_sequence=self.current_sequence,
                next_sequence=next_sequence,
                mutations=next_mutations,
                prev_fitness=self.current_fitness,
                step=self.step_count,
                is_feasible=is_feasible,
                violations=violations,
            )
            self.evaluations_count += 1
            if self.memory is not None:
                var_record = VariantRecord(
                    variant_id=f"RL_STEP_{self.step_count}_{self.current_fitness:.2f}",
                    protein_id="DESIGN",
                    mutations=next_mutations,
                    mutated_sequence=next_sequence,
                    raw_score=breakdown["current_fitness"],
                    is_feasible=is_feasible,
                    feasibility_violations=violations,
                )
                step_ind = Individual(
                    variant=var_record,
                    fitness_values={
                        "predicted_fitness": breakdown["current_fitness"],
                        "esm2_plausibility": breakdown["esm_score"],
                        "mutational_parsimony": float(-len(next_mutations)),
                    },
                    objectives_vector=[
                        breakdown["current_fitness"],
                        breakdown["esm_score"],
                        float(-len(next_mutations)),
                    ],
                    is_evaluated=True,
                )
                self.memory.register_candidate(step_ind)

        # Update state
        self.current_sequence = next_sequence
        self.accumulated_mutations = next_mutations
        self.modified_positions.add(pos)
        self.current_fitness = breakdown["current_fitness"]
        self.current_esm_score = breakdown["esm_score"]
        self.step_count += 1

        terminated = False
        truncated = (self.step_count >= self.max_mutations)

        obs = self._get_observation()
        info = {
            "sequence": self.current_sequence,
            "mutations": [m.model_dump() for m in self.accumulated_mutations],
            "fitness": self.current_fitness,
            "esm_score": self.current_esm_score,
            "reward_breakdown": breakdown,
            "action_mask": self.get_action_mask() if not (terminated or truncated) else None,
        }
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Constructs standardized observation feature vector for Actor-Critic network."""
        # 1. One-hot representation of active mutable positions
        one_hot = np.zeros(self.seq_feature_dim, dtype=np.float32)
        for i, pos in enumerate(self.mutable_positions):
            res = self.current_sequence[pos]
            aa_idx = self.aa_to_idx.get(res, 0)
            one_hot[i * len(self.amino_acids) + aa_idx] = 1.0

        # 2. Scalar state metadata
        mut_ratio = float(len(self.accumulated_mutations) / max(1, self.max_mutations))
        rem_steps = float((self.max_mutations - self.step_count) / max(1, self.max_mutations))
        fit_norm = float(self.current_fitness / 10.0)
        esm_norm = float(self.current_esm_score / 10.0)

        scalars = np.array([mut_ratio, rem_steps, fit_norm, esm_norm], dtype=np.float32)
        return np.concatenate([one_hot, scalars])
