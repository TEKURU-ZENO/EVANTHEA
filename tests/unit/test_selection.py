"""Unit tests for parent selection mechanisms (Tournament, Roulette, Random)."""

import pytest

from evanthea.data.schemas import VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.evolution.population import Population
from evanthea.evolution.selection import (
    RandomSelector,
    RouletteWheelSelector,
    TournamentSelector,
)


def _ind(v_id: str, fit: float, rank: int = 0, cd: float = 0.0) -> Individual:
    v = VariantRecord(variant_id=v_id, protein_id="P", mutations=[], mutated_sequence="A", is_wildtype=True)
    return Individual(
        variant=v,
        fitness_values={"fit": fit},
        objectives_vector=[fit],
        rank=rank,
        crowding_distance=cd,
        is_evaluated=True,
    )


def test_tournament_selector_single_objective():
    i1 = _ind("low", fit=1.0)
    i2 = _ind("high", fit=10.0)
    pop = Population(individuals=[i1, i2])

    selector = TournamentSelector(tournament_size=2, use_crowding=False, single_objective_name="fit", seed=42)
    selected = selector.select(pop, n_select=10)
    assert len(selected) == 10
    # Higher fitness individual should win all 2-individual tournaments
    assert all(ind.id == "high" for ind in selected)


def test_tournament_selector_crowded_comparison():
    # Rank 0 beats Rank 1
    i_r0 = _ind("r0", fit=5.0, rank=0, cd=1.0)
    i_r1 = _ind("r1", fit=8.0, rank=1, cd=5.0)
    pop = Population(individuals=[i_r0, i_r1])

    selector = TournamentSelector(tournament_size=2, use_crowding=True, seed=42)
    selected = selector.select(pop, n_select=5)
    assert all(ind.id == "r0" for ind in selected)


def test_roulette_wheel_selector():
    i1 = _ind("low", fit=1.0)
    i2 = _ind("huge", fit=100.0)
    pop = Population(individuals=[i1, i2])

    selector = RouletteWheelSelector(objective_name="fit", seed=123)
    selected = selector.select(pop, n_select=20)
    assert len(selected) == 20
    # Huge individual should be selected with overwhelming probability (>90%)
    huge_count = sum(1 for ind in selected if ind.id == "huge")
    assert huge_count >= 16


def test_random_selector_and_empty_population():
    i1 = _ind("a", fit=1.0)
    i2 = _ind("b", fit=2.0)
    pop = Population(individuals=[i1, i2])

    selector = RandomSelector(seed=42)
    selected = selector.select(pop, n_select=4)
    assert len(selected) == 4

    empty_pop = Population()
    with pytest.raises(ValueError):
        selector.select(empty_pop, n_select=1)

    with pytest.raises(ValueError):
        TournamentSelector(tournament_size=0)
