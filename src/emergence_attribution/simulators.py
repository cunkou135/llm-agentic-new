"""Instrumented simulators with counter-based matched random streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from .dsl import compute_indicator
from .raw_schemas import raw_schema
from .reference_truth import (
    controlled_structural_contexts,
    mechanism_target_for_variant,
    reference_processes,
    reference_relations,
)


def counter_rng(seed: int, step: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(step), int(stream)]))


def _group_grid(agent_grid: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.full(agent_grid.shape, -1, dtype=np.int8)
    occupied = agent_grid >= 0
    result[occupied] = groups[agent_grid[occupied]]
    return result


def _district_grid(
    height: int, width: int, district_rows: int, district_columns: int
) -> np.ndarray:
    """Return a fixed rectangular district partition for any grid dimensions."""

    if not 1 <= district_rows <= height or not 1 <= district_columns <= width:
        raise ValueError("district layout must fit within the grid")
    row_ids = np.minimum(
        np.arange(height, dtype=np.int32) * district_rows // height,
        district_rows - 1,
    )
    column_ids = np.minimum(
        np.arange(width, dtype=np.int32) * district_columns // width,
        district_columns - 1,
    )
    return (
        row_ids[:, None] * district_columns + column_ids[None, :]
    ).astype(np.int16)


def _neighbour_values(group_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupied = group_grid >= 0
    same = np.zeros(group_grid.shape, dtype=np.int16)
    total = np.zeros(group_grid.shape, dtype=np.int16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbour = np.roll(np.roll(group_grid, dx, axis=0), dy, axis=1)
            valid = neighbour >= 0
            total += valid
            same += valid & occupied & (neighbour == group_grid)
    similarity = np.divide(same, total, out=np.ones_like(same, dtype=float), where=total > 0)
    return similarity, same, total


def _destination_similarity(group_grid: np.ndarray, flat_index: int, group: int) -> float:
    height, width = group_grid.shape
    row, column = divmod(int(flat_index), width)
    values = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            value = group_grid[(row + dx) % height, (column + dy) % width]
            if value >= 0:
                values.append(int(value == group))
    return float(np.mean(values)) if values else 0.5


def _toroidal_distance(
    origin: tuple[int, int], destination: tuple[int, int], height: int, width: int
) -> float:
    dy = min(abs(origin[0] - destination[0]), height - abs(origin[0] - destination[0]))
    dx = min(abs(origin[1] - destination[1]), width - abs(origin[1] - destination[1]))
    return float(dx + dy)


def _delayed_channel(
    driver: np.ndarray,
    structural_context: np.ndarray,
    lag: int,
    sign: int,
    seed: int,
    stream: int,
    autoregression: float,
    disabled: bool,
) -> np.ndarray:
    driver = np.asarray(driver, dtype=float)
    structural_context = np.asarray(structural_context, dtype=float)
    if driver.shape != structural_context.shape:
        raise ValueError("controlled driver and structural context must have equal shape")
    output = np.empty(len(driver), dtype=float)
    output[0] = 0.5
    for time in range(1, len(driver)):
        innovation = counter_rng(seed, time, stream).normal(0.0, 0.025)
        context = float(np.clip(structural_context[time], 0.0, 1.0)) - 0.5
        target = 0.5 + 0.30 * context + innovation
        if not disabled and time >= lag:
            centred = float(np.clip(driver[time - lag], 0.0, 1.0)) - 0.5
            target += 0.60 * sign * centred
        output[time] = np.clip(
            autoregression * output[time - 1] + (1.0 - autoregression) * target,
            0.0,
            1.0,
        )
    return output


def _attach_controlled_channels(
    scenario: str,
    raw: dict[str, np.ndarray],
    seed: int,
    mechanism_variant: str,
) -> dict[str, np.ndarray]:
    processes = {item.process_id: item for item in reference_processes(scenario)}
    relations = reference_relations(scenario)
    micro_to_meso = relations[:]
    prefix = "s" if scenario == "schelling" else "d"
    meso = np.empty((int(raw["num_steps"][0]), 4), dtype=float)
    macro = np.empty_like(meso)
    disabled_name = mechanism_target_for_variant(scenario, mechanism_variant)
    contexts = controlled_structural_contexts(scenario)
    for index in range(4):
        edge = next(item for item in micro_to_meso if item.target == f"{prefix}_meso_{index}")
        process = processes[edge.source]
        driver = compute_indicator(
            process.computation,
            process.temporal_aggregation,
            raw,
            raw_schema(scenario),
        )
        meso_context = compute_indicator(
            contexts[index][0],
            {"op": "identity"},
            raw,
            raw_schema(scenario),
        )
        macro_context = compute_indicator(
            contexts[index][1],
            {"op": "identity"},
            raw,
            raw_schema(scenario),
        )
        meso[:, index] = _delayed_channel(
            driver,
            meso_context,
            edge.lag,
            edge.sign,
            seed,
            700 + index,
            0.45,
            edge.mechanism == disabled_name,
        )
        second = next(item for item in relations if item.source == f"{prefix}_meso_{index}")
        macro[:, index] = _delayed_channel(
            meso[:, index],
            macro_context,
            second.lag,
            second.sign,
            seed,
            800 + index,
            0.70,
            second.mechanism == disabled_name,
        )
    return {"mechanism_channel": np.column_stack([meso, macro]).astype(np.float32)}


@dataclass(frozen=True)
class SchellingParameters:
    tolerance: float
    move_probability: float
    destination_preference: float


def simulate_schelling(
    seed: int,
    spec: dict[str, Any],
    parameters: dict[str, float],
    mechanism_variant: str = "baseline",
) -> dict[str, np.ndarray]:
    mechanism_target_for_variant("schelling", mechanism_variant)
    steps = int(spec["num_steps"])
    agents = int(spec["num_agents"])
    height, width = int(spec["grid_height"]), int(spec["grid_width"])
    if agents >= height * width:
        raise ValueError("the grid must contain at least one vacancy")
    p = SchellingParameters(
        float(parameters["tolerance"]),
        float(parameters["move_probability"]),
        float(parameters["destination_preference"]),
    )
    preference = 0.0 if mechanism_variant == "disable_homophilic_relocation" else p.destination_preference
    districts = _district_grid(
        height,
        width,
        int(spec.get("district_rows", 3)),
        int(spec.get("district_columns", 3)),
    )
    initial = counter_rng(seed, 0, 1)
    groups = np.arange(agents, dtype=np.int8) % 2
    initial.shuffle(groups)
    occupied = initial.choice(height * width, size=agents, replace=False)
    agent_grid = np.full(height * width, -1, dtype=np.int32)
    agent_grid[occupied] = np.arange(agents, dtype=np.int32)
    agent_grid = agent_grid.reshape(height, width)
    positions = np.column_stack(np.unravel_index(occupied, (height, width))).astype(np.int32)
    raw: dict[str, np.ndarray] = {
        "state_grid": np.empty((steps, height, width), dtype=np.int8),
        "agent_id": np.arange(agents, dtype=np.int32),
        "agent_group": groups,
        "agent_position": np.empty((steps, agents, 2), dtype=np.int32),
        "district_id": np.empty((steps, agents), dtype=np.int16),
        "local_similarity": np.empty((steps, agents), dtype=np.float32),
        "neighbour_count": np.empty((steps, agents), dtype=np.int16),
        "unhappy": np.zeros((steps, agents), dtype=bool),
        "unhappy_count": np.zeros(steps, dtype=np.int32),
        "moved": np.zeros((steps, agents), dtype=bool),
        "move_distance": np.zeros((steps, agents), dtype=np.float32),
        "destination_similarity": np.zeros((steps, agents), dtype=np.float32),
        "boundary_agent": np.zeros((steps, agents), dtype=bool),
        "num_steps": np.asarray([steps], dtype=np.int32),
        "agent_count": np.asarray([agents], dtype=np.int32),
    }
    max_distance = max(height // 2 + width // 2, 1)
    for time in range(steps):
        group_grid = _group_grid(agent_grid, groups)
        raw["state_grid"][time] = group_grid
        raw["agent_position"][time] = positions
        similarity_grid, same_grid, total_grid = _neighbour_values(group_grid)
        rows, columns = positions[:, 0], positions[:, 1]
        raw["district_id"][time] = districts[rows, columns]
        similarity = similarity_grid[rows, columns]
        total = total_grid[rows, columns]
        same = same_grid[rows, columns]
        unhappy = (total > 0) & (similarity < p.tolerance)
        boundary = (total > 0) & (same < total)
        raw["local_similarity"][time] = similarity
        raw["neighbour_count"][time] = total
        raw["unhappy"][time] = unhappy
        raw["unhappy_count"][time] = int(np.sum(unhappy))
        raw["boundary_agent"][time] = boundary
        activation = counter_rng(seed, time, 10).random(agents)
        moving = np.flatnonzero(unhappy & (activation < p.move_probability))
        order_key = counter_rng(seed, time, 11).random(agents)
        moving = moving[np.argsort(order_key[moving])]
        for agent in moving:
            vacancies = np.flatnonzero(agent_grid.ravel() < 0)
            if not len(vacancies):
                break
            rng = counter_rng(seed, time, 1000 + int(agent))
            candidates = rng.choice(vacancies, size=min(6, len(vacancies)), replace=False)
            scores = np.asarray(
                [_destination_similarity(group_grid, int(item), int(groups[agent])) for item in candidates]
            )
            if rng.random() < preference:
                destination_flat = int(candidates[int(np.argmax(scores))])
            else:
                destination_flat = int(candidates[0])
            origin = tuple(int(value) for value in positions[agent])
            destination = divmod(destination_flat, width)
            agent_grid[origin] = -1
            agent_grid[destination] = int(agent)
            positions[agent] = destination
            group_grid[origin] = -1
            group_grid[destination] = groups[agent]
            raw["moved"][time, agent] = True
            raw["move_distance"][time, agent] = _toroidal_distance(
                origin, destination, height, width
            ) / max_distance
            raw["destination_similarity"][time, agent] = _destination_similarity(
                group_grid, destination_flat, int(groups[agent])
            )
    return raw


@dataclass(frozen=True)
class DeffuantParameters:
    confidence_bound: float
    assimilation_strength: float
    backfire_threshold: float
    backfire_strength: float


def _opinion_network(seed: int, agents: int, degree: int, rewire: float) -> np.ndarray:
    graph_seed = int(np.random.SeedSequence([seed, 0, 40]).generate_state(1)[0])
    graph = nx.watts_strogatz_graph(agents, degree, rewire, seed=graph_seed)
    return np.asarray(sorted((min(a, b), max(a, b)) for a, b in graph.edges()), dtype=np.int32)


def _network_neighbours(edges: np.ndarray, agents: int) -> list[np.ndarray]:
    neighbours: list[list[int]] = [[] for _ in range(agents)]
    for left, right in np.asarray(edges, dtype=np.int32):
        neighbours[int(left)].append(int(right))
        neighbours[int(right)].append(int(left))
    result = [np.asarray(sorted(values), dtype=np.int32) for values in neighbours]
    if any(not len(values) for values in result):
        raise RuntimeError("adaptive interaction network contains an isolated agent")
    return result


def _rewire_interaction_edge(
    edge_set: set[tuple[int, int]],
    focal: int,
    partner: int,
    opinions: np.ndarray,
    random_priority: np.ndarray,
    prefer_homophily: bool,
) -> bool:
    """Replace one sampled tie while preserving a simple undirected graph."""

    old_edge = (min(focal, partner), max(focal, partner))
    if old_edge not in edge_set:
        return False
    partner_degree = sum(partner in edge for edge in edge_set)
    if partner_degree <= 1:
        # Rewiring is a tie replacement, not node deletion.  Keeping the
        # previous tie avoids creating an isolated agent that could no longer
        # participate in the next step's neighbour-sampling rule.
        return False
    current_neighbours = {
        right if left == focal else left
        for left, right in edge_set
        if left == focal or right == focal
    }
    candidates = sorted(
        set(range(len(opinions))) - {focal, partner} - current_neighbours
    )
    if not candidates:
        return False
    ordered = sorted(candidates, key=lambda node: (float(random_priority[node]), node))
    sampled = ordered[: min(6, len(ordered))]
    if prefer_homophily:
        destination = min(
            sampled,
            key=lambda node: (
                abs(float(opinions[focal]) - float(opinions[node])),
                float(random_priority[node]),
                node,
            ),
        )
    else:
        destination = sampled[0]
    new_edge = (min(focal, destination), max(focal, destination))
    if new_edge in edge_set or new_edge[0] == new_edge[1]:
        return False
    edge_set.remove(old_edge)
    edge_set.add(new_edge)
    return True


def simulate_deffuant(
    seed: int,
    spec: dict[str, Any],
    parameters: dict[str, float],
    mechanism_variant: str = "baseline",
) -> dict[str, np.ndarray]:
    mechanism_target_for_variant("deffuant", mechanism_variant)
    steps, agents = int(spec["num_steps"]), int(spec["num_agents"])
    edges = _opinion_network(
        seed, agents, int(spec["network_degree"]), float(spec["network_rewire_probability"])
    )
    edge_count = len(edges)
    adaptive_rewiring_probability = float(
        spec.get("adaptive_rewiring_probability", 0.15)
    )
    rewiring_homophily_probability = float(
        spec.get("rewiring_homophily_probability", 0.65)
    )
    if not 0.0 <= adaptive_rewiring_probability <= 1.0:
        raise ValueError("adaptive_rewiring_probability must be within [0, 1]")
    if not 0.0 <= rewiring_homophily_probability <= 1.0:
        raise ValueError("rewiring_homophily_probability must be within [0, 1]")
    p = DeffuantParameters(
        float(parameters["confidence_bound"]),
        float(parameters["assimilation_strength"]),
        float(parameters["backfire_threshold"]),
        0.0 if mechanism_variant == "disable_backfire" else float(parameters["backfire_strength"]),
    )
    initial = counter_rng(seed, 0, 41)
    component = initial.integers(0, 2, size=agents)
    opinions = np.clip(
        np.where(component == 0, -0.35, 0.35) + initial.normal(0.0, 0.18, agents),
        -1.0,
        1.0,
    )
    raw: dict[str, np.ndarray] = {
        "state_opinion": np.empty((steps, agents), dtype=np.float32),
        "network_edges": np.empty((steps, edge_count, 2), dtype=np.int32),
        "partner_id": np.empty((steps, agents), dtype=np.int32),
        "interaction_distance": np.empty((steps, agents), dtype=np.float32),
        "interaction_accepted": np.zeros((steps, agents), dtype=bool),
        "interaction_backfire": np.zeros((steps, agents), dtype=bool),
        "interaction_rejected": np.zeros((steps, agents), dtype=bool),
        "edge_rewired": np.zeros((steps, agents), dtype=bool),
        "agent_shift": np.zeros((steps, agents), dtype=np.float32),
        "sign_flip": np.zeros((steps, agents), dtype=bool),
        "extreme_agent_count": np.zeros(steps, dtype=np.int32),
        "num_steps": np.asarray([steps], dtype=np.int32),
        "agent_count": np.asarray([agents], dtype=np.int32),
    }
    for time in range(steps):
        raw["state_opinion"][time] = opinions
        raw["network_edges"][time] = edges
        neighbours = _network_neighbours(edges, agents)
        uniforms = counter_rng(seed, time, 42).random(agents)
        partners = np.asarray(
            [
                neighbours[index][min(int(uniforms[index] * len(neighbours[index])), len(neighbours[index]) - 1)]
                for index in range(agents)
            ],
            dtype=np.int32,
        )
        difference = opinions[partners] - opinions
        distance = np.abs(difference)
        accepted = distance <= p.confidence_bound
        backfire = (~accepted) & (distance >= p.backfire_threshold) & (p.backfire_strength > 0)
        rejected = ~(accepted | backfire)
        shift = np.zeros(agents, dtype=float)
        shift[accepted] = p.assimilation_strength * difference[accepted]
        shift[backfire] = (
            p.backfire_strength
            * np.sign(opinions[backfire] - opinions[partners[backfire]])
            * np.minimum(distance[backfire], 1.0)
        )
        updated = np.clip(opinions + shift, -1.0, 1.0)
        raw["partner_id"][time] = partners
        raw["interaction_distance"][time] = distance
        raw["interaction_accepted"][time] = accepted
        raw["interaction_backfire"][time] = backfire
        raw["interaction_rejected"][time] = rejected
        raw["agent_shift"][time] = shift
        raw["sign_flip"][time] = np.sign(updated) != np.sign(opinions)
        raw["extreme_agent_count"][time] = int(np.sum(np.abs(opinions) >= 0.75))
        rewiring_draws = counter_rng(seed, time, 43).random(agents)
        homophily_draws = counter_rng(seed, time, 44).random(agents)
        edge_set = {
            (int(left), int(right)) for left, right in np.asarray(edges, dtype=np.int32)
        }
        for agent in range(agents):
            if not (rejected[agent] or backfire[agent]):
                continue
            if rewiring_draws[agent] >= adaptive_rewiring_probability:
                continue
            priority = counter_rng(seed, time, 4000 + agent).random(agents)
            raw["edge_rewired"][time, agent] = _rewire_interaction_edge(
                edge_set,
                agent,
                int(partners[agent]),
                opinions,
                priority,
                bool(homophily_draws[agent] < rewiring_homophily_probability),
            )
        edges = np.asarray(sorted(edge_set), dtype=np.int32)
        if len(edges) != edge_count:
            raise RuntimeError("adaptive rewiring changed the network edge count")
        opinions = updated
    return raw


def simulate_toy(seed: int, steps: int = 80, agents: int = 12) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(steps, agents))
    y = np.zeros(steps, dtype=float)
    for time in range(2, steps):
        y[time] = 0.7 * y[time - 1] + 0.8 * np.mean(x[time - 2]) + rng.normal(0, 0.1)
    return {"x": x, "y": y}


def run_scenario(
    scenario: str,
    seed: int,
    spec: dict[str, Any],
    parameters: dict[str, float],
    mechanism_variant: str = "baseline",
) -> dict[str, np.ndarray]:
    if scenario == "schelling":
        return simulate_schelling(seed, spec, parameters, mechanism_variant)
    if scenario == "deffuant":
        return simulate_deffuant(seed, spec, parameters, mechanism_variant)
    raise KeyError(f"unknown scenario: {scenario}")


def run_scenario_with_hidden(
    scenario: str,
    seed: int,
    spec: dict[str, Any],
    parameters: dict[str, float],
    mechanism_variant: str = "baseline",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run a simulator and return physically separate public and hidden payloads."""

    public = run_scenario(scenario, seed, spec, parameters, mechanism_variant)
    hidden = _attach_controlled_channels(
        scenario=scenario,
        raw=public,
        seed=seed,
        mechanism_variant=mechanism_variant,
    )
    if set(public) & set(hidden):
        raise RuntimeError("public and hidden simulator payloads overlap")
    return public, hidden
