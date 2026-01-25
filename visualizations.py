import pandas as pd
import matplotlib.pyplot as plt
import re
import json
import os

def generate_metric_tables(json_data, base_filename):
    all_methods = set()
    for prob in json_data.values():
        all_methods.update(prob['solvers'].keys())
    
    def get_sort_key(method_name):
        if method_name == "PGD":
            return (0, 0)
        if method_name == "Newton":
            return (1, 0)
        if "MN" in method_name:
            match = re.search(r'm(\d+)', method_name)
            if match:
                return (2, int(match.group(1)))
        return (3, method_name)

    methods = sorted(list(all_methods), key=get_sort_key)
    
    tables = {"time": [], "iter": [], "obj": []}

    for prob_name, data in json_data.items():
        n_vars = data['n']
        prob_label = f"{prob_name.replace('_', '\\_')} ({n_vars})"
        
        row_time = {"Problem (N)": prob_label}
        row_iter = {"Problem (N)": prob_label}
        row_obj  = {"Problem (N)": prob_label}
        
        solvers = data['solvers']

        valid_times = [s['time'] for s in solvers.values() if s.get('time', 0) > 0]
        min_time = min(valid_times) if valid_times else None
        
        valid_f = [s['f'] for s in solvers.values()]
        min_obj = min(valid_f) if valid_f else None

        valid_iters = [s['iterations'] for s in solvers.values() if s.get('iterations', 0) > 0]
        min_iter = min(valid_iters) if valid_iters else None

        for m in methods:
            res = solvers.get(m)
            if res is None:
                row_time[m] = row_iter[m] = row_obj[m] = "-"
                continue

            t_val = res.get('time', 0)
            f_val = res.get('f', 0)
            i_tot = res.get('iterations', 0)
            i_newt = res.get('newton_iterations', 0)

            t_str = f"{t_val:.4f}"
            if min_time is not None and t_val <= min_time:
                t_str = f"\\textbf{{{t_str}}}"
            row_time[m] = t_str

            if m == "PGD":
                i_str = f"{i_tot}"
            else:
                i_str = f"{i_tot} ({i_newt})"
            
            if min_iter is not None and i_tot <= min_iter:
                i_str = f"\\textbf{{{i_str}}}"
            row_iter[m] = i_str

            f_str = f"{f_val:.2e}"
            if min_obj is not None and f_val <= min_obj + 1e-13:
                f_str = f"\\textbf{{{f_str}}}"
            row_obj[m] = f_str

        tables["time"].append(row_time)
        tables["iter"].append(row_iter)
        tables["obj"].append(row_obj)

    for metric, rows in tables.items():
        df = pd.DataFrame(rows)
        col_fmt = "l" + "c" * (len(df.columns) - 1)
        
        latex_tabular = df.to_latex(index=False, escape=False, column_format=col_fmt)
        
        full_latex = (
            "\\begin{table}[h!]\n"
            "\\centering\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            f"{latex_tabular}"
            "}\n"
            f"\\caption{{Metric comparison: {metric.replace('_', ' ').capitalize()}}}\n"
            "\\end{table}"
        )
        
        with open(f"{base_filename.split('.')[0]}_{metric}.tex", "w") as f:
            f.write(full_latex)

def plot_scalability(json_data, problem_base_name):
    
    sub_data = {k: v for k, v in json_data.items() if k.startswith(problem_base_name)}
    
    sorted_keys = sorted(sub_data.keys(), key=lambda x: int(re.search(r'N(\d+)', x).group(1)))
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-paper')
    
    solvers = list(sub_data[sorted_keys[0]]['solvers'].keys())

    for solver in solvers:
        n_values = []
        times = []
        for key in sorted_keys:
            n_values.append(sub_data[key]['n'])
            times.append(sub_data[key]['solvers'][solver]['time'])
        
        plt.plot(n_values, times, marker='o', label=solver, linewidth=1.5)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.title(f"Scalability: {problem_base_name}", fontsize=14)
    plt.xlabel("Problem Size (n)", fontsize=12)
    plt.ylabel("Execution Time (s)", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/scalability_{problem_base_name}.png", bbox_inches='tight')


json_regular = json.load(open("data/full_benchmark_other_regular.json"))
generate_metric_tables(json_regular, "tables/results_regular.csv")

json_alternative = json.load(open("data/full_benchmark_alternative.json"))
generate_metric_tables(json_alternative, "tables/results_alternative.csv")

varying_n_json = json.load(open("data/full_benchmark_varying_n.json"))

for p_base in ["HADAMALS", "CHARDIS0"]:
    plot_scalability(varying_n_json, p_base)