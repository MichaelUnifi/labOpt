import pycutest
import numpy as np
from scipy.optimize import minimize, Bounds
from utils import projected_gradient_descent, project, projected_newton, modified_projected_newton
from tqdm import tqdm
import json
from time import perf_counter

def run_benchmark(problem_name, solvers_to_test, sif_params=None):
    p = pycutest.import_problem(problem_name, sifParams=sif_params)
    
    x0, lb, ub = p.x0, p.bl, p.bu
    results = {"n": len(x0), "solvers": {}}

    for solver_name, solver_func in solvers_to_test.items():
        start_time = perf_counter()
        res = solver_func(p.obj, p.grad, p.hess, x0, lb, ub)
        end_time = perf_counter()
        
        res["time"] = end_time - start_time
        results["solvers"][solver_name] = res
    
    return results

problem_sets = {
    "other_regular": ["BRANIN", "DGOSPEC", "HS38", "HS45", "HART6", "S368", "HOLMES", "HADAMALS", "PROBPENL", "SINEALI"],
    "alternative": ["JNLBRNG1", "HS110", "BQPGASIM"],
    "other_varying_n": ["HADAMALS", "CHARDIS0"],
}

varying_n_configs = {
    "HADAMALS": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 32],
    "CHARDIS0": [5, 9, 20, 30, 50, 100, 200, 500, 1000, 2000],
}

solvers = {
    "PGD": lambda f, g, h, x, l, u: projected_gradient_descent(f, g, x, l, u),
    "Newton": lambda f, g, h, x, l, u: projected_newton(f, g, h, x, l, u)
}

for m_val in range(5, 41, 5):
    solvers[f"MN-m{m_val}"] = lambda f, g, h, x, l, u, m=m_val: \
        modified_projected_newton(f, g, h, x, m, l, u)

all_results = {}
current_set = "other_regular"

for prob in tqdm(problem_sets[current_set]):
    print(f"Benchmarking {prob}...")
    all_results[prob] = run_benchmark(prob, solvers)
with open(f"data/full_benchmark_{current_set}.json", "w") as f:
    json.dump(all_results, f, indent=4)

alt_results = {}
current_set = "alternative"

for prob in tqdm(problem_sets[current_set]):
    print(f"Benchmarking {prob}...")
    alt_results[prob] = run_benchmark(prob, solvers)
with open(f"data/full_benchmark_{current_set}.json", "w") as f:
    json.dump(alt_results, f, indent=4)

all_varying_results = {}

for prob_name, n_list in varying_n_configs.items():
    for n in tqdm(n_list, desc=f"Testing {prob_name}"):
        test_id = f"{prob_name}_N{n}"
        
        if prob_name == "CHARDIS0":
            sif_params = {'NP1': n}
        elif prob_name == "POWELLBC":
            sif_params = {'P': n}
        else:
            sif_params = {'N': n}
        all_varying_results[test_id] = run_benchmark(
            prob_name, 
            solvers, 
            sif_params=sif_params
        )

with open("data/full_benchmark_varying_n.json", "w") as f:
    json.dump(all_varying_results, f, indent=4)