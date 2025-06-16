import time
import os
import numpy as np
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution

from input import ChallengeSolution, ChallengeSolver, Challenge, compute_objective, log_results

MAX_ITER = int(10e4)
TIME_LIMIT = 595

cplex_log_file = os.path.join("analysis","parametric_cplex.log")

def heuristic_warm(orders, aisles, wave_size_lb, wave_size_ub):
    # Heurística para estimar t inicial
    rng = np.random.default_rng(42)
    selected_orders = []
    total_items_heur = 0

    # Seleciona pedidos aleatórios até atingir o limite inferior
    order_indices = rng.permutation(len(orders))
    for o in order_indices:
        order_items = sum(orders[o].values())
        if total_items_heur + order_items <= wave_size_ub:
            selected_orders.append(o)
            total_items_heur += order_items
            if total_items_heur >= wave_size_lb:
                break

    # Determina corredores necessários
    selected_aisles = set()
    for o in selected_orders:
        for i, q in orders[o].items():
            for a in range(len(aisles)):
                if aisles[a].get(i):
                    selected_aisles.add(a)

    # Estima razão e inicializa t
    if selected_aisles:
        t = total_items_heur / len(selected_aisles)
        print(f"Heuristic init: total_items = {total_items_heur}, aisles = {len(selected_aisles)}, t = {t:.4f}")
    else:
        t = 0.0

    return t, selected_orders, selected_aisles


class DOcplexChallengeSolver(ChallengeSolver):

    def solve(self, start_time) -> ChallengeSolution:
        n_items = self.n_items
        orders: list[dict, int] = self.orders
        aisles: list[dict, int] = self.aisles

        n_orders = len(orders)
        n_aisles = len(aisles)


        t, selected_orders, selected_aisles = heuristic_warm(orders, aisles, self.wave_size_lb, self.wave_size_ub)

        best_orders = []
        best_aisles = []
        best_ratio = 0.0
        tolerance = 1e-4
        max_iterations = MAX_ITER

        # Construir modelo base
        base_model = Model(name="WavePicking", log_output=False)
        x = base_model.binary_var_list(n_orders, name="x")
        y = base_model.binary_var_list(n_aisles, name="y")
        total_items_expr = base_model.sum(sum(orders[o].values()) * x[o] for o in range(n_orders))     

        for i in range(n_items):
            expr = base_model.sum(orders[o].get(i) * x[o] for o in range(n_orders) if orders[o].get(i)) - \
                   base_model.sum(aisles[a].get(i) * y[a] for a in range(n_aisles) if aisles[a].get(i))
            base_model.add_constraint(expr <= 0)

        base_model.add_constraint(total_items_expr >= self.wave_size_lb)
        base_model.add_constraint(total_items_expr <= self.wave_size_ub)
        base_model.parameters.mip.tolerances.mipgap = 0.01

        # Cria solução inicial para warm start
        x_vals_init = [1 if i in selected_orders else 0 for i in range(n_orders)]
        y_vals_init = [1 if j in selected_aisles else 0 for j in range(n_aisles)]
        prev_solution = (x_vals_init, y_vals_init)

        for iter in range(max_iterations):
            remaining_time = TIME_LIMIT - (time.perf_counter() - start_time)
            if remaining_time <= 5:
                print("Timeout approaching, returning best solution found so far.")
                break

            print(f"--- Iteration {iter+1}, t = {t:.4f}, remaining time = {remaining_time:.2f} s ---")
            mdl = base_model.clone()

            # Função objetivo parametrizada
            obj_expr = total_items_expr - mdl.sum(t * y[a] for a in range(n_aisles))
            mdl.set_objective("max", obj_expr)

            # Warm start
            if prev_solution is not None:
                for i, val in enumerate(prev_solution[0]):
                    mdl.get_var_by_name(f"x_{i}").start = val
                for j, val in enumerate(prev_solution[1]):
                    mdl.get_var_by_name(f"y_{j}").start = val

            with open(cplex_log_file, '+a') as f:
                f.write(f'\nt = {t}\n')
                solution: SolveSolution = mdl.solve(log_output=f, time_limit=remaining_time)

            if not solution:
                print("No feasible solution found.")
                break

            x_vals = [solution.get_value(f"x_{i}") for i in range(n_orders)]
            y_vals = [solution.get_value(f"y_{j}") for j in range(n_aisles)]
            total_items = sum(sum(orders[o].values()) * x_vals[o] for o in range(n_orders))
            total_aisles = np.sum(y_vals)

            # print("Orders:", x_vals)
            # print("Aisles:", y_vals)

            if total_aisles == 0:
                break

            new_ratio = total_items / total_aisles

            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_orders = [i for i, v in enumerate(x_vals) if v >= 0.5]
                best_aisles = [j for j, v in enumerate(y_vals) if v >= 0.5]

            if abs(new_ratio - t) < tolerance:
                print("Convergence reached.")
                break

            # Atualiza t e salva solução atual para warm start
            t = new_ratio
            prev_solution = (x_vals, y_vals)

        elapsed = time.perf_counter() - start_time
        print(f"\nDOcplex solved in {elapsed:.2f} seconds. Best ratio: {best_ratio:.4f}")

        if best_orders and best_aisles:
            return ChallengeSolution(best_orders, best_aisles)
        
        best_aisles = np.ones(n_aisles)
        best_orders = [np.random.randint(0, n_orders)]
        return ChallengeSolution(best_orders, best_aisles)

if __name__ == "__main__":
    import sys

    try:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        log_file = sys.argv[3]
    except:
        input_file = "datasets/b/instance_0011.txt"
        output_file = "output.txt"
        log_file = os.path.join("analysis","parametric_results.log")
        
    with open(cplex_log_file, '+a') as f:
        f.write('='*80)
        f.write('\n')
        f.write(input_file)

    start = time.perf_counter()

    challenge = Challenge()
    challenge.read_input(input_file)

    solver = DOcplexChallengeSolver(challenge.orders, challenge.aisles, challenge.n_items,
                                    challenge.wave_size_lb, challenge.wave_size_ub)

    solution = solver.solve(start)
    challenge.write_output(solution, output_file)

    elapsed_time = time.perf_counter() - start
    objective = compute_objective(challenge.orders, solution)

    log_results(log_file, input_file, elapsed_time, objective)
