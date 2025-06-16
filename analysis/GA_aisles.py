import tempfile
import time
import shutil
import numpy as np
import os
import cplex
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.termination import get_termination


from input import ChallengeSolution, ChallengeSolver, Challenge, compute_objective, log_results

POP_SIZE = 32
TIME_LIMIT = 590

cplex_time_log = os.path.join("analysis/cplex_times.txt")

def model_max_items(n_orders, u_oi, u_ai, n_items, selected_aisles, 
                    wave_size_lb, wave_size_ub, start_time,
                    best_ratio
                    ):
    start_time_model = time.perf_counter()

    cpx = cplex.Cplex()
    # cpx.set_log_stream(None)          # desabilita saída padrão
    # cpx.set_results_stream(None)      # desabilita saída padrão
    cpx.set_warning_stream(None)

    # Variáveis binárias x[o] para cada pedido o
    x_names = [f"x_{o}" for o in range(n_orders)]
    cpx.variables.add(names=x_names, types=[cpx.variables.type.binary]*n_orders)

    # Objetivo: maximizar total de itens pedidos na onda
    # total_items_expr = sum(o) sum(i) u_oi[o,i] * x[o]
    # Como u_oi é matriz (n_orders x n_items), podemos calcular o coeficiente para cada x[o] somando sobre i

    obj_coefs = np.sum(u_oi, axis=1)
    cpx.objective.set_sense(cpx.objective.sense.maximize)
    cpx.objective.set_linear(list(zip(x_names, obj_coefs)))

    # Restrição: wave_size_lb <= total_items <= wave_size_ub
    # total_items = sum o sum i u_oi[o,i] x[o] = sum o (sum i u_oi[o,i]) x[o]
    # Igual a objetivo

    ind = list(range(n_orders))
    val = obj_coefs
    # Lower bound constraint
    cpx.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=x_names, val=val)],
        senses=["G"],
        rhs=[wave_size_lb],
        names=["wave_size_lb"]
    )
    # Upper bound constraint
    cpx.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=x_names, val=val)],
        senses=["L"],
        rhs=[wave_size_ub],
        names=["wave_size_ub"]
    )

    # Restrição para capacidade total por item nas aisles selecionadas:
    # sum o u_oi[o,i] * x[o] <= total_item_capacity[i]  para cada item i
    total_item_capacity = np.sum(u_ai[selected_aisles], axis=0)

    for i in range(n_items):
        # coeficientes para x[o] nesta restrição são u_oi[o,i]
        coef = [u_oi[o, i] for o in range(n_orders)]
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=x_names, val=coef)],
            senses=["L"],
            rhs=[total_item_capacity[i]],
            names=[f"cap_item_{i}"]
        )

    # Definir limite de tempo
    time_limit = TIME_LIMIT - (time.perf_counter() - start_time)
    if time_limit > 1:
        cpx.parameters.timelimit.set(time_limit)

        best_num_items = best_ratio * len(selected_aisles)
        cpx.parameters.mip.tolerances.lowercutoff = best_num_items
    else:
        raise TimeoutError("Tempo máximo!")

    # Resolver e salvar log em arquivo
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp_name = tmp.name

        # Write initial message to temp file
        print("\n->Solving new subproblem...", file=tmp)
        tmp.flush()  # Ensure content is written before redirecting streams

        # Redirect CPLEX output streams to the temp file
        with open(tmp_name, "a") as f_log:
            cpx.set_log_stream(None)
            cpx.set_results_stream(None)
            cpx.set_warning_stream(None)
            cpx.solve()
            print("Subproblem solving time:", time.perf_counter() - start_time_model, file=f_log)

    # Append temp file contents to your real log
    with open(tmp_name, "r") as tmp, open(cplex_time_log, "a") as final_log:
        shutil.copyfileobj(tmp, final_log)

    if cpx.solution.is_primal_feasible():
        solution_vals = cpx.solution.get_values()
        selected_orders = [o for o, val in enumerate(solution_vals) if val > 0.5]
        obj_val = cpx.solution.get_objective_value()
        return obj_val, selected_orders
    else:
        return 0, None

class HybridGA(ElementwiseProblem):
    def __init__(self, n_aisles, n_orders, n_items, u_oi, u_ai, wave_size_lb, wave_size_ub, start_time,
                 **kwargs):
        self.n_aisles = n_aisles
        self.n_orders = n_orders
        self.n_items = n_items
        self.u_oi = u_oi
        self.u_ai = u_ai
        self.wave_size_lb = wave_size_lb
        self.wave_size_ub = wave_size_ub
        self.start_time = start_time

        self.best_aisles = np.ones(n_aisles)
        self.best_orders = [np.random.randint(0, n_orders)]
        self.best_ratio = np.sum(u_oi[self.best_orders], axis=1) / n_aisles


        super().__init__(n_var=n_aisles, n_obj=1, xl=0, xu=1, vtype=bool,**kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        if time.perf_counter() - self.start_time > TIME_LIMIT:
            raise TimeoutError("Tempo máximo!")
        
        selected_aisles = x.astype(bool)
        if not np.any(selected_aisles):
            out["F"] = 1e6
            return

        total_items, selected_orders = model_max_items(self.n_orders, self.u_oi, self.u_ai, self.n_items,
                                       selected_aisles, self.wave_size_lb, self.wave_size_ub,
                                       self.start_time, self.best_ratio)
        num_aisles = np.sum(selected_aisles)

        if num_aisles == 0:
            out["F"] = 1e6
        else:
            ratio = total_items / num_aisles
            
            if ratio > self.best_ratio:
                print(ratio)
                self.best_orders = selected_orders
                self.best_ratio = ratio
                self.best_aisles = x

            out["F"] = -ratio  # Minimizar o negativo da métrica de interesse

class NoImprovementTimeLimitTermination(TimeBasedTermination):
    def __init__(self, max_time, n_gen_no_improve=10):
        super().__init__(max_time)

        self.n_gen_no_improve = n_gen_no_improve
        self.best_fitness = None
        self.counter = 0

    def _do_continue(self, algorithm):
        
        current_best_list = algorithm.opt.get("F")
        if current_best_list is None or len(current_best_list) == 0:
            # No solution yet, so continue running
            return True

        current_best = current_best_list[0]
        if self.best_fitness is None:
            self.best_fitness = current_best
            self.counter = 0
            return True

        if current_best < self.best_fitness:  # minimizing
            self.best_fitness = current_best
            self.counter = 0
        else:
            self.counter += 1        

        return self.counter < self.n_gen_no_improve and not self.has_terminated()


    def _do_finalize(self, algorithm):
        print()

class HybridGAChallengeSolver(ChallengeSolver):
    def solve(self, start_time) -> ChallengeSolution:
        n_orders = len(self.orders)
        n_aisles = len(self.aisles)
        n_items = self.n_items

        u_oi = np.zeros((n_orders, n_items))
        u_ai = np.zeros((n_aisles, n_items))

        for o in range(n_orders):
            for i, q in self.orders[o].items():
                u_oi[o, i] = float(q)

        for a in range(n_aisles):
            for i, q in self.aisles[a].items():
                u_ai[a, i] = float(q)


        n_threads = 8
        pool = ThreadPool(n_threads)
        runner = StarmapParallelization(pool.starmap)

        problem = HybridGA(n_aisles, n_orders, n_items, u_oi, u_ai, 
                           self.wave_size_lb, self.wave_size_ub, start_time,
                           elementwise_runner=runner
                           )

        algorithm = GA(
            POP_SIZE,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True)
        
        try:
            # Create termination objects
            max_time = TIME_LIMIT - (time.perf_counter() - start_time)
            termination = NoImprovementTimeLimitTermination(max_time, n_gen_no_improve=10)
            if max_time < 0:
                raise TimeoutError('Tempo máximo!')
            
            res = minimize(problem,
                        algorithm,
                        termination,
                        seed=43,
                        verbose=True)
        except TimeoutError as e:
            print(str(e))
        except Exception as e:
            print(str(e))
            raise e

        selected_aisles = list(np.where(problem.best_aisles)[0])
        selected_orders = problem.best_orders

        return ChallengeSolution(orders=selected_orders, aisles=selected_aisles)

if __name__ == "__main__":
    import sys

    try:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        log_file = sys.argv[3]
    except:
        input_file = "datasets/a/instance_0001.txt"
        output_file = "output.txt"
        log_file = "analysis/GA_analysis.log"

    with open(cplex_time_log, "a") as f:
        print("="*50, file=f)
        print(input_file, file=f)

    start = time.perf_counter()

    challenge = Challenge()
    challenge.read_input(input_file)


    solver = HybridGAChallengeSolver(challenge.orders, challenge.aisles, challenge.n_items,
                                    challenge.wave_size_lb, challenge.wave_size_ub)

    solution = solver.solve(start)
    challenge.write_output(solution, output_file)

    elapsed_time = time.perf_counter() - start
    objective = compute_objective(challenge.orders, solution)

    log_results(log_file, input_file, elapsed_time, objective)
    with open(cplex_time_log, "a") as f:
        print()