"""
Benchmark: standard multi-objective GA vs distributional GA (distribution-level Pareto + particle Pareto)
on a synthetic breeding optimization task.

Objectives
---------
1) Maximize breeding value (EBV)
2) Minimize mean kinship (proxy for inbreeding)

We overlay the Pareto sets produced by:
- Standard GA with native multi-objective support
- Distributional GA sweeping scalarization weights to approximate a Pareto set
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from evosolve import train_sel, train_sel_control
from evosolve.core import TrainSelResult


def make_breeding_data(n_animals: int = 40, kinship_strength: float = 0.25, seed: int = 7):
    """Create synthetic breeding values and a positive semidefinite kinship matrix."""
    rng = np.random.default_rng(seed)
    breeding_values = rng.normal(loc=1.0, scale=0.25, size=n_animals)

    # Random PSD kinship, scaled to realistic range
    A = rng.normal(size=(n_animals, n_animals))
    kinship = A @ A.T
    kinship /= kinship.max()  # scale to [0, 1]
    kinship *= kinship_strength  # dampen to typical inbreeding range

    return {
        "breeding_values": breeding_values,
        "kinship": kinship,
    }


def compute_objectives(selection: np.ndarray, data: dict) -> Tuple[float, float]:
    """Return (breeding value, -mean kinship) for maximization."""
    bv = float(np.mean(data["breeding_values"][selection]))
    kmat = data["kinship"]
    k = len(selection)
    if k <= 1:
        mean_kinship = 0.0
    else:
        sub = kmat[np.ix_(selection, selection)]
        mean_kinship = float((sub.sum() - np.trace(sub)) / (k * (k - 1)))
    return bv, -mean_kinship


def parse_selection(int_vals, n_animals: int = None) -> np.ndarray:
    """Convert TrainSel int_vals payload into a 1D integer selection array."""
    base = int_vals[0] if isinstance(int_vals, (list, tuple)) else int_vals
    selection = np.array(base, dtype=int).ravel()
    if selection.ndim == 0:
        selection = selection.reshape(1)
    if n_animals is not None:
        selection = np.clip(selection, 0, n_animals - 1)
    return selection


def breeding_stat(int_vals, data=None, dbl_vals=None):
    """Two-objective evaluator for standard GA."""
    selection = parse_selection(int_vals)
    bv, neg_kin = compute_objectives(selection, data)
    return [bv, neg_kin]


def make_scalarized_stat(weight_bv: float):
    """Create scalarized objective for distributional GA (weight_bv in [0, 1])."""
    weight_bv = float(weight_bv)
    weight_kin = 1.0 - weight_bv

    def _stat(int_vals, dbl_vals, data):
        selection = parse_selection(int_vals)
        bv, neg_kin = compute_objectives(selection, data)
        # Normalize scales so kinship is comparable to EBV
        return weight_bv * bv + weight_kin * neg_kin

    return _stat


def run_standard_ga(data: dict, select_k: int, n_animals: int) -> TrainSelResult:
    """Run standard multi-objective GA to get Pareto set."""
    control = train_sel_control(
        niterations=100,
        npop=350,
        nelite=140,
        nEliteSaved=100,
        mutprob=0.25,
        mutintensity=0.2,
        crossprob=0.8,
        crossintensity=0.5,
        nislands=2,
        niterIslands=50,
    )

    return train_sel(
        candidates=[list(range(n_animals))],
        setsizes=[select_k],
        settypes=["UOS"],
        stat=breeding_stat,
        n_stat=2,
        data=data,
        control=control,
        verbose=False,
    )


def run_distributional_ga_multiobjective(data: dict, select_k: int, n_animals: int):
    """
    Run distributional GA directly in multi-objective mode using weighted evaluation.

    Returns
    -------
    TrainSelResult
        Contains pareto_front with aggregated objectives per distribution.
    """
    control = train_sel_control(
        niterations=200,
        npop=300,
        nelite=100,
        nEliteSaved=40,
        mutprob=0.4,
        mutintensity=0.35,
        crossprob=0.85,
        crossintensity=0.6,
        use_cma_es=False,  # CMA-ES incompatible with distributional
        # Distributional-specific
        dist_objective="cvar",  # HV (particle set) + entropy bonus
        dist_tau=0.4,
        dist_lambda_var=0.1,
        dist_eval_mode="weighted",  # weighted scalarization
        dist_n_samples=30,
        dist_K_particles=50,
        dist_weight_mutation_prob=0.7,
        dist_support_mutation_prob=0.9,
        dist_birth_death_prob=0.4,
        dist_birth_rate=0.25,
        dist_death_rate=0.25,
        dist_alpha=0.1,
        dist_hv_ref=(-1, 1),
        dist_use_nsga_means=True,
    )

    return train_sel(
        candidates=[list(range(n_animals))],
        setsizes=[select_k],
        settypes=["DIST:UOS"],
        stat=breeding_stat,  # two-objective evaluator
        n_stat=2,
        data=data,
        control=control,
        verbose=False,
    )


def pareto_filter(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return non-dominated points for maximization in (bv, neg_kin) space."""
    kept = []
    for i, (bv_i, kin_i) in enumerate(points):
        dominated = False
        for j, (bv_j, kin_j) in enumerate(points):
            if j == i:
                continue
            if (bv_j >= bv_i and kin_j >= kin_i) and (bv_j > bv_i or kin_j > kin_i):
                dominated = True
                break
        if not dominated:
            kept.append((bv_i, kin_i))
    return kept


def extract_particle_objectives(dist_result: TrainSelResult, data: dict, n_animals: int) -> List[Tuple[float, float]]:
    """
    Pull particle-level objectives from distributional Pareto solutions.

    Returns
    -------
    List[(float, float)]
        (bv, neg_kin) for each particle across all Pareto distributions.
    """
    points: List[Tuple[float, float]] = []
    if not dist_result.pareto_solutions:
        return points
    for entry in dist_result.pareto_solutions:
        dist = entry.get("distribution")
        if dist is None:
            continue
        for p in getattr(dist, "particles", []):
            sel = parse_selection(p.int_values[0], n_animals)
            points.append(compute_objectives(sel, data))
    return points


def sample_distribution_objectives(dist_result: TrainSelResult, data: dict, n_animals: int, per_dist_samples: int = 100) -> List[Tuple[float, float]]:
    """
    Monte Carlo sample objectives from each Pareto distribution for visualization.

    Parameters
    ----------
    per_dist_samples : int
        Number of samples per distributional Pareto solution
    """
    points: List[Tuple[float, float]] = []
    sources = []
    if dist_result.pareto_solutions:
        sources.extend(dist_result.pareto_solutions)
    elif getattr(dist_result, "distribution", None) is not None:
        sources.append({"distribution": dist_result.distribution})
    else:
        return points
    rng = np.random.default_rng(42)
    for entry in sources:
        dist = entry.get("distribution")
        if dist is None or not getattr(dist, "particles", None):
            continue
        # sample indices respecting weights
        weights = getattr(dist, "weights", np.ones(len(dist.particles)) / len(dist.particles))
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()
        idx = rng.choice(len(dist.particles), size=per_dist_samples, p=weights)
        for i in idx:
            p = dist.particles[i]
            sel = parse_selection(p.int_values[0], n_animals)
            points.append(compute_objectives(sel, data))
    return points


def sample_distribution_solutions(
    dist_result: TrainSelResult,
    data: dict,
    n_animals: int,
    per_dist_samples: int = 10,
    seed: int = 123
) -> List[dict]:
    """
    Draw samples from each optimized distribution to inspect selections post-optimization.

    Returns
    -------
    List[dict]
        Each entry has keys: 'selection' (np.ndarray) and 'objectives' (bv, -kinship).
    """
    samples: List[dict] = []
    rng = np.random.default_rng(seed)
    sources = []
    if dist_result.pareto_solutions:
        sources.extend(dist_result.pareto_solutions)
    elif getattr(dist_result, "distribution", None) is not None:
        sources.append({"distribution": dist_result.distribution})
    else:
        return samples

    for entry in sources:
        dist = entry.get("distribution")
        if dist is None or not getattr(dist, "particles", None):
            continue
        weights = getattr(dist, "weights", np.ones(len(dist.particles)) / len(dist.particles))
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()
        idx = rng.choice(len(dist.particles), size=per_dist_samples, p=weights)
        for i in idx:
            p = dist.particles[i]
            sel = parse_selection(p.int_values[0], n_animals)
            obj = compute_objectives(sel, data)
            samples.append({"selection": sel, "objectives": obj})
    return samples


def compute_distribution_mean_objective(dist, data: dict, n_animals: int) -> Tuple[float, float]:
    """Compute weighted mean objectives of a distribution using its particles."""
    particles = getattr(dist, "particles", [])
    weights = getattr(dist, "weights", None)
    if not particles or weights is None:
        return (0.0, 0.0)
    objs = []
    for p in particles:
        sel = parse_selection(p.int_values[0], n_animals)
        objs.append(compute_objectives(sel, data))
    arr = np.array(objs, dtype=float)
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    return tuple(np.sum(arr * w[:, None], axis=0))


def plot_overlay(
    std_front: List[Tuple[float, float]],
    dist_front: List[Tuple[float, float]],
    dist_particle_front: List[Tuple[float, float]] = None,
    dist_sample_points: List[Tuple[float, float]] = None,
    dist_means_raw: List[Tuple[float, float]] = None
):
    """Plot and save overlay of Pareto fronts (solutions and particle-level)."""
    std_arr = np.array(std_front)
    dist_arr = np.array(dist_front)
    part_arr = np.array(dist_particle_front) if dist_particle_front else None
    sample_arr = np.array(dist_sample_points) if dist_sample_points else None
    dist_raw_arr = np.array(dist_means_raw) if dist_means_raw else None

    plt.figure(figsize=(8, 5))
    plt.scatter(-std_arr[:, 1], std_arr[:, 0], c="#1f77b4", label="Standard GA Pareto", alpha=0.75)
    plt.scatter(-dist_arr[:, 1], dist_arr[:, 0], c="#d62728", marker="x", label="Distribution means", alpha=0.8, s=70)
    if dist_raw_arr is not None and len(dist_raw_arr) > 0:
        plt.scatter(-dist_raw_arr[:, 1], dist_raw_arr[:, 0], c="#d62728", alpha=0.08, s=18, label="_nolegend_")
    if part_arr is not None and len(part_arr) > 0:
        plt.scatter(-part_arr[:, 1], part_arr[:, 0], c="#ff7f0e", s=26, alpha=0.6, label="Particles (Pareto)")
    if sample_arr is not None and len(sample_arr) > 0:
        plt.scatter(-sample_arr[:, 1], sample_arr[:, 0], c="#ffbf80", s=8, alpha=0.12, label="_nolegend_")

    plt.xlabel("Mean kinship (lower is better)")
    plt.ylabel("Mean breeding value (higher is better)")
    plt.title("Standard vs Distributional GA on Breeding Pareto Benchmark")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("examples/distributional_vs_standard_pareto.png", dpi=200)
    print("Saved overlay plot to examples/distributional_vs_standard_pareto.png")


def main():
    n_animals = 40
    select_k = 12
    data = make_breeding_data(n_animals=n_animals, kinship_strength=0.2, seed=11)

    # Run standard GA (multi-objective)
    std_result = run_standard_ga(data, select_k, n_animals)
    std_front = [(pf[0], pf[1]) for pf in std_result.pareto_front]

    # Run distributional GA in multi-objective mode (weighted evaluation)
    dist_result = run_distributional_ga_multiobjective(data, select_k, n_animals)
    # Recompute true objectives for distributions/particles to avoid optimizer-specific aggregation noise
    dist_mean_points: List[Tuple[float, float]] = []
    dist_mean_points_raw: List[Tuple[float, float]] = []
    if dist_result.pareto_solutions:
        for entry in dist_result.pareto_solutions:
            dist = entry.get("distribution")
            if dist is None:
                continue
            obj = compute_distribution_mean_objective(dist, data, n_animals)
            dist_mean_points.append(obj)
            dist_mean_points_raw.append(obj)
    dist_front = pareto_filter(dist_mean_points) if dist_mean_points else []

    particle_points = extract_particle_objectives(dist_result, data, n_animals)
    particle_front = pareto_filter(particle_points) if particle_points else []
    sampled_points = sample_distribution_objectives(dist_result, data, n_animals, per_dist_samples=400)
    sampled_solutions = sample_distribution_solutions(dist_result, data, n_animals, per_dist_samples=5, seed=21)

    plot_overlay(std_front, dist_front, particle_front, sampled_points, dist_mean_points_raw)

    print(f"Standard GA Pareto size: {len(std_front)}")
    print(f"Distributional GA (distribution means) Pareto size: {len(dist_front)}")
    print(f"Distributional GA (particle Pareto) size: {len(particle_front)}")
    if sampled_solutions:
        preview = sampled_solutions[:3]
        print("Sampled selections from optimized distributions (first 3):")
        for entry in preview:
            sel_str = np.array2string(entry["selection"], separator=",")
            obj = entry["objectives"]
            print(f"  sel={sel_str} -> (bv={obj[0]:.3f}, -kin={obj[1]:.3f})")


if __name__ == "__main__":
    main()
