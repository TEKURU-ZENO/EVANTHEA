"""Memory-informed evolutionary strategies: evaluation bypassing, archive elitism, and novelty filters."""

from typing import Dict, List, Optional, Tuple

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator
from evanthea.memory.manager import EvolutionaryMemory
from evanthea.protein.sequence import compute_sequence_hash


class EvaluationBypasser:
    """Intercepts candidates to retrieve cached evaluations from memory, eliminating redundant compute."""

    def __init__(self, memory: EvolutionaryMemory) -> None:
        self.memory = memory

    def partition_and_hydrate(
        self,
        individuals: List[Individual],
    ) -> Tuple[List[Individual], List[Individual]]:
        """Partitions individuals into cached (hydrated from memory) and novel (requiring evaluation).

        Also handles intra-batch duplicates so only unique novel candidates are marked for evaluation.

        Args:
            individuals: List of individuals to be evaluated.

        Returns:
            Tuple of (bypassed_individuals, novel_representatives).
        """
        bypassed: List[Individual] = []
        novel_by_hash: Dict[str, List[Individual]] = {}
        novel_representatives: List[Individual] = []

        for ind in individuals:
            if ind.is_evaluated:
                bypassed.append(ind)
                continue

            cached_rec = self.memory.lookup_exact(ind.sequence)
            if cached_rec is not None and cached_rec.fitness_values:
                # Cache hit: assign cached scores with zero forward passes
                ind.fitness_values = dict(cached_rec.fitness_values)
                ind.objectives_vector = list(cached_rec.objectives_vector) if cached_rec.objectives_vector else None
                ind.is_evaluated = True
                bypassed.append(ind)
            else:
                seq_hash = compute_sequence_hash(ind.sequence)
                if seq_hash in novel_by_hash:
                    # Intra-batch duplicate
                    novel_by_hash[seq_hash].append(ind)
                    self.memory._stats.total_proposals += 1
                    self.memory._stats.cache_hits += 1
                    self.memory._stats.evaluations_saved += 1
                    bypassed.append(ind)
                else:
                    novel_by_hash[seq_hash] = [ind]
                    novel_representatives.append(ind)

        return bypassed, novel_representatives

    def evaluate_with_memory(
        self,
        individuals: List[Individual],
        evaluator: MultiObjectiveEvaluator,
    ) -> int:
        """Evaluates only novel individuals while hydrating cached ones. Returns count of new evaluations."""
        novel_by_hash: Dict[str, List[Individual]] = {}
        novel_representatives: List[Individual] = []

        for ind in individuals:
            if ind.is_evaluated:
                continue

            cached_rec = self.memory.lookup_exact(ind.sequence)
            if cached_rec is not None and cached_rec.fitness_values:
                # Cache hit: assign cached scores with zero forward passes
                ind.fitness_values = dict(cached_rec.fitness_values)
                ind.objectives_vector = list(cached_rec.objectives_vector) if cached_rec.objectives_vector else None
                ind.is_evaluated = True
            else:
                seq_hash = compute_sequence_hash(ind.sequence)
                if seq_hash in novel_by_hash:
                    # Intra-batch duplicate
                    novel_by_hash[seq_hash].append(ind)
                    self.memory._stats.total_proposals += 1
                    self.memory._stats.cache_hits += 1
                    self.memory._stats.evaluations_saved += 1
                else:
                    novel_by_hash[seq_hash] = [ind]
                    novel_representatives.append(ind)

        if novel_representatives:
            evaluator.evaluate_population(novel_representatives)
            for rep in novel_representatives:
                self.memory.register_candidate(rep)
                # Hydrate any intra-batch duplicates sharing this sequence
                seq_hash = compute_sequence_hash(rep.sequence)
                for dup in novel_by_hash.get(seq_hash, []):
                    if dup is not rep:
                        dup.fitness_values = dict(rep.fitness_values)
                        dup.objectives_vector = list(rep.objectives_vector) if rep.objectives_vector else None
                        dup.is_evaluated = True

        return len(novel_representatives)


class ArchiveElitismInjector:
    """Maintains global non-dominated archive and injects historical elites to restore population diversity."""

    def __init__(self, memory: EvolutionaryMemory, max_injection: int = 3) -> None:
        self.memory = memory
        self.max_injection = max_injection

    def inject_archive_elites(
        self,
        population_individuals: List[Individual],
        target_size: int,
        generation: int,
    ) -> List[Individual]:
        """Injects up to max_injection global Pareto candidates if population has available capacity."""
        if len(population_individuals) >= target_size:
            return population_individuals

        needed = min(target_size - len(population_individuals), self.max_injection)
        if needed <= 0:
            return population_individuals

        archive_recs = self.memory.get_global_pareto_archive()
        existing_seqs = {ind.sequence for ind in population_individuals}

        injected = list(population_individuals)
        count = 0

        for r in archive_recs:
            if r.sequence not in existing_seqs and count < needed:
                v = VariantRecord(
                    variant_id=r.variant_id,
                    protein_id="GFP",
                    mutations=r.mutations,
                    mutated_sequence=r.sequence,
                    is_wildtype=len(r.mutations) == 0,
                    metadata={"origin": "archive_injection"},
                )
                ind = Individual(
                    variant=v,
                    fitness_values=dict(r.fitness_values),
                    objectives_vector=list(r.objectives_vector) if r.objectives_vector else None,
                    is_evaluated=True,
                    generation=generation,
                    parent_ids=r.parent_ids,
                )
                injected.append(ind)
                existing_seqs.add(r.sequence)
                count += 1

        return injected
