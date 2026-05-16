import argparse
import math
import inspect
import os
import random
import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from sko.GA import GA_TSP
from sko.PSO import PSO


DATA_DIR = "data"
OUTPUT_DIR = "outputs"
TSPLIB_URLS = {
    "berlin52": [
        "https://raw.githubusercontent.com/pdrozdowski/TSPLib.Net/master/TSPLIB95/tsp/berlin52.tsp",
        "https://raw.githubusercontent.com/mastqe/tsplib/master/instances/berlin52.tsp",
        "https://raw.githubusercontent.com/coin-or/tsplib/master/tsp/berlin52.tsp",
        "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/berlin52.tsp",
    ],
}


@dataclass
class RunResult:
    best_length: float
    history: List[float]
    elapsed_sec: float
    peak_mem_bytes: int


def download_tsplib(name: str, target_path: str) -> None:
    urls = TSPLIB_URLS.get(name)
    if not urls:
        raise ValueError(f"Unknown TSPLIB instance: {name}")
    last_error = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(target_path, "wb") as f:
                f.write(resp.content)
            return
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(f"Failed to download {name} from known mirrors") from last_error


def load_tsp_coords(path: str) -> np.ndarray:
    coords = []
    in_section = False
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("NODE_COORD_SECTION"):
                in_section = True
                continue
            if line.startswith("EOF"):
                break
            if in_section:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    if not coords:
        raise ValueError("No coordinates found in TSP file")
    return np.array(coords, dtype=np.float64)


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    dist = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d = int(round(math.hypot(dx, dy)))
            dist[i, j] = d
            dist[j, i] = d
    return dist


def tour_length(route: np.ndarray, dist: np.ndarray) -> int:
    total = 0
    n = route.shape[0]
    for i in range(n):
        total += dist[route[i - 1], route[i]]
    return total


def tournament_select(pop: List[np.ndarray], fitness: np.ndarray, k: int) -> np.ndarray:
    idx = np.random.choice(len(pop), size=k, replace=False)
    best = idx[np.argmax(fitness[idx])]
    return pop[best]


def order_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    n = p1.shape[0]
    cut = random.randint(1, n - 2)
    child_prefix = list(p1[:cut])
    remaining = [x for x in p2 if x not in child_prefix]
    return np.array(child_prefix + remaining, dtype=np.int32)


def swap_mutation(route: np.ndarray) -> None:
    i, j = random.sample(range(route.shape[0]), 2)
    route[i], route[j] = route[j], route[i]


def ga_tsp_custom(
    dist: np.ndarray,
    pop_size: int,
    max_iter: int,
    crossover_rate: float,
    mutation_rate: float,
    seed: int,
) -> RunResult:
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    pop = [rng.permutation(n).astype(np.int32) for _ in range(pop_size)]
    history = []
    tracemalloc.start()
    start = time.perf_counter()

    for _ in range(max_iter):
        lengths = np.array([tour_length(p, dist) for p in pop], dtype=np.float64)
        fitness = 1.0 / lengths
        best_idx = int(np.argmin(lengths))
        history.append(float(lengths[best_idx]))

        new_pop = [pop[best_idx].copy()]
        while len(new_pop) < pop_size:
            parent1 = tournament_select(pop, fitness, k=3)
            parent2 = tournament_select(pop, fitness, k=3)
            child = parent1.copy()
            if random.random() < crossover_rate:
                child = order_crossover(parent1, parent2)
            if random.random() < mutation_rate:
                swap_mutation(child)
            new_pop.append(child)
        pop = new_pop

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    best_length = min(history)
    return RunResult(best_length, history, elapsed, peak)


def pso_tsp_custom(
    dist: np.ndarray,
    pop_size: int,
    max_iter: int,
    w: float,
    c1: float,
    c2: float,
    vmax: float,
    seed: int,
) -> RunResult:
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    x = rng.random((pop_size, n))
    v = rng.uniform(-vmax, vmax, size=(pop_size, n))
    pbest = x.copy()
    pbest_len = np.array([tour_length(np.argsort(p), dist) for p in pbest], dtype=np.float64)
    gbest_idx = int(np.argmin(pbest_len))
    gbest = pbest[gbest_idx].copy()

    history = []
    tracemalloc.start()
    start = time.perf_counter()

    for _ in range(max_iter):
        r1 = rng.random((pop_size, n))
        r2 = rng.random((pop_size, n))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -vmax, vmax)
        x = x + v
        x = np.clip(x, 0.0, 1.0)

        lengths = np.array([tour_length(np.argsort(p), dist) for p in x], dtype=np.float64)
        improved = lengths < pbest_len
        pbest[improved] = x[improved]
        pbest_len[improved] = lengths[improved]
        gbest_idx = int(np.argmin(pbest_len))
        gbest = pbest[gbest_idx].copy()
        history.append(float(pbest_len[gbest_idx]))

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    best_length = min(history)
    return RunResult(best_length, history, elapsed, peak)


def ga_tsp_library(
    dist: np.ndarray,
    pop_size: int,
    max_iter: int,
    crossover_rate: float,
    mutation_rate: float,
    seed: int,
) -> RunResult:
    n = dist.shape[0]

    def objective(route: np.ndarray) -> float:
        return float(tour_length(route, dist))

    random.seed(seed)
    np.random.seed(seed)
    tracemalloc.start()
    start = time.perf_counter()

    ga_kwargs = {
        "func": objective,
        "n_dim": n,
        "size_pop": pop_size,
        "max_iter": max_iter,
        "prob_mut": mutation_rate,
        "prob_cros": crossover_rate,
    }
    ga_sig = inspect.signature(GA_TSP.__init__)
    ga_filtered = {k: v for k, v in ga_kwargs.items() if k in ga_sig.parameters}
    ga = GA_TSP(**ga_filtered)
    best_x, best_y = ga.run()

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    best_y_val = float(np.asarray(best_y).reshape(-1)[0])
    history = getattr(ga, "best_y_history", [])
    if len(history) == 0:
        history = [best_y_val]
    history_vals = [float(np.asarray(x).reshape(-1)[0]) for x in history]
    return RunResult(best_y_val, history_vals, elapsed, peak)


def pso_tsp_library(
    dist: np.ndarray,
    pop_size: int,
    max_iter: int,
    w: float,
    c1: float,
    c2: float,
    seed: int,
) -> RunResult:
    n = dist.shape[0]

    def objective(x: np.ndarray) -> float:
        route = np.argsort(x)
        return float(tour_length(route, dist))

    np.random.seed(seed)
    tracemalloc.start()
    start = time.perf_counter()

    pso = PSO(
        func=objective,
        dim=n,
        pop=pop_size,
        max_iter=max_iter,
        w=w,
        c1=c1,
        c2=c2,
        lb=0.0,
        ub=1.0,
    )
    best_x, best_y = pso.run()

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    best_y_val = float(np.asarray(best_y).reshape(-1)[0])
    history = getattr(pso, "gbest_y_hist", [])
    if len(history) == 0:
        history = [best_y_val]
    history_vals = [float(np.asarray(x).reshape(-1)[0]) for x in history]
    return RunResult(best_y_val, history_vals, elapsed, peak)


def ensure_tsp_path(path: str, name: str) -> str:
    if path:
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, f"{name}.tsp")
    if not os.path.exists(local_path):
        download_tsplib(name, local_path)
    return local_path


def save_convergence(histories: dict, max_iter: int) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for key, history in histories.items():
        series = history[:max_iter]
        df = pd.DataFrame({"iter": np.arange(1, len(series) + 1), "best": series})
        df.to_csv(os.path.join(OUTPUT_DIR, f"{key}_convergence.csv"), index=False)

    plt.figure(figsize=(8, 5))
    for key, history in histories.items():
        plt.plot(history[:max_iter], label=key)
    plt.xlabel("Iteration")
    plt.ylabel("Best Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "convergence.png"), dpi=150)
    plt.close()


def summarize_results(results: dict) -> pd.DataFrame:
    rows = []
    for name, runs in results.items():
        best_vals = [r.best_length for r in runs]
        times = [r.elapsed_sec for r in runs]
        mems = [r.peak_mem_bytes for r in runs]
        rows.append(
            {
                "Method": name,
                "BestMean": float(np.mean(best_vals)),
                "BestStd": float(np.std(best_vals)),
                "AvgTimeSec": float(np.mean(times)),
                "PeakMemMB": float(np.mean(mems)) / (1024 * 1024),
            }
        )
    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame) -> None:
    path = os.path.join(OUTPUT_DIR, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsp", default="", help="Path to TSPLIB .tsp file")
    parser.add_argument("--name", default="berlin52", help="TSPLIB instance name")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--crossover", type=float, default=0.8)
    parser.add_argument("--mutation", type=float, default=0.2)
    parser.add_argument("--w", type=float, default=0.8)
    parser.add_argument("--c1", type=float, default=1.5)
    parser.add_argument("--c2", type=float, default=1.5)
    parser.add_argument("--vmax", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tsp_path = ensure_tsp_path(args.tsp, args.name)
    coords = load_tsp_coords(tsp_path)
    dist = compute_distance_matrix(coords)

    results = {
        "custom_ga": [],
        "library_ga": [],
        "custom_pso": [],
        "library_pso": [],
    }

    for run_id in range(args.runs):
        seed = args.seed + run_id
        results["custom_ga"].append(
            ga_tsp_custom(
                dist,
                args.pop,
                args.iters,
                args.crossover,
                args.mutation,
                seed,
            )
        )
        results["library_ga"].append(
            ga_tsp_library(
                dist,
                args.pop,
                args.iters,
                args.crossover,
                args.mutation,
                seed,
            )
        )
        results["custom_pso"].append(
            pso_tsp_custom(
                dist,
                args.pop,
                args.iters,
                args.w,
                args.c1,
                args.c2,
                args.vmax,
                seed,
            )
        )
        results["library_pso"].append(
            pso_tsp_library(
                dist,
                args.pop,
                args.iters,
                args.w,
                args.c1,
                args.c2,
                seed,
            )
        )

    histories = {
        "custom_ga": results["custom_ga"][0].history,
        "library_ga": results["library_ga"][0].history,
        "custom_pso": results["custom_pso"][0].history,
        "library_pso": results["library_pso"][0].history,
    }
    save_convergence(histories, args.iters)

    summary = summarize_results(results)
    write_summary(summary)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
