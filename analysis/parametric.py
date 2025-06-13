import time
import numpy as np
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution

from input import ChallengeSolution, ChallengeSolver, Challenge, compute_objective, log_results

MAX_ITER = 30
TIME_LIMIT = 590

class DOcplexChallengeSolver(ChallengeSolver):

    def solve(self, start_time) -> ChallengeSolution:
        n_orders = len(self.orders)
        n_aisles = len(self.aisles)
        n_items = self.n_items

        # Preprocessamento
        u_oi = np.zeros((n_orders, n_items), dtype=int)
        u_ai = np.zeros((n_aisles, n_items), dtype=int)
        for o in range(n_orders):
            for i, q in self.orders[o].items():
                u_oi[o, i] = float(q)
        for a in range(n_aisles):
            for i, q in self.aisles[a].items():
                u_ai[a, i] = float(q)

        best_orders = []
        best_aisles = []
        best_ratio = 0.0
        tolerance = 1e-4
        max_iterations = MAX_ITER
        t = 0.0

        # Construir modelo base
        base_model = Model(name="WavePicking", log_output=False)
        x = base_model.binary_var_list(n_orders, name="x")
        y = base_model.binary_var_list(n_aisles, name="y")
        total_items_expr = base_model.sum(np.sum(u_oi[o]) * x[o] for o in range(n_orders))


        

        for i in range(n_items):
            expr = base_model.sum(u_oi[o][i] * x[o] for o in range(n_orders) if u_oi[o][i] > 0) - \
                   base_model.sum(u_ai[a][i] * y[a] for a in range(n_aisles) if u_ai[a][i] > 0)
            base_model.add_constraint(expr <= 0)

        base_model.add_constraint(total_items_expr >= self.wave_size_lb)
        base_model.add_constraint(total_items_expr <= self.wave_size_ub)

        prev_solution = None

        for iter in range(max_iterations):
            remaining_time = TIME_LIMIT - (time.perf_counter() - start_time)
            if remaining_time <= 1:
                print("Timeout approaching, returning best solution found so far.")
                break

            print(f"\n--- Iteration {iter+1}, t = {t:.4f}, remaining time = {remaining_time:.2f} s ---")
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

            solution: SolveSolution = mdl.solve(log_output=False, time_limit=remaining_time)
            if not solution:
                print("No feasible solution found.")
                break

            x_vals = [solution.get_value(f"x_{i}") for i in range(n_orders)]
            y_vals = [solution.get_value(f"y_{j}") for j in range(n_aisles)]
            total_items = sum(np.sum(u_oi[o]) * x_vals[o] for o in range(n_orders))
            total_aisles = sum(y_vals)

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
        return None


if __name__ == "__main__":
    import sys

    try:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        log_file = sys.argv[3]
    except:
        input_file = "datasets/a/instance_0001.txt"
        output_file = "output.txt"
        log_file = "analysis.log"

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
