# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 14:23:06 2025

@author: felip
"""

import cplex
from input_ import *
import time 
import logging

class Approche1(ChallengeSolver):
    def __init__(self, orders, aisles, n_items, wave_lb, wave_ub):
        
        super().__init__(orders, aisles, n_items, wave_lb, wave_ub)
        
        
        # Get weight for each order
        self.p_j = {i: sum(self.orders[i].values())  for i in range(len(self.orders))}
        self.n_orders = len(self.orders)
        self.n_aisles = len(self.aisles)
        
        self.order_name = [f"x{i}" for i in range(self.n_orders)]
        self.aisle_name = [f"y{i}" for i in range(self.n_aisles)]
        
        self.time_limit = 600
        self.model = None
        self.last_solution = None
        self.warm_start = True
        self.best_sol = None

    

    def write_model(self) -> ChallengeSolution:
        # Solution phase
        model = cplex.Cplex()
        
        model.objective.set_sense(model.objective.sense.maximize)
        x_obj = list(self.p_j.values())
        
        model.variables.add(names=self.order_name,
                        types=[model.variables.type.binary]*self.n_orders,
                        obj=x_obj)
        
        model.variables.add(names=self.aisle_name,
                        types=[model.variables.type.binary]*self.n_aisles,
                        obj=[0]*self.n_aisles)
        
        model.variables.add(
            names=["n_open_aisles"],
            types=[model.variables.type.integer],  
            obj=[0] 
        )
        
        coeffs = [1] * self.n_aisles

        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=self.aisle_name + ["n_open_aisles"],
                                       val=coeffs + [-1])],
            senses=["E"],
            rhs=[0],
            names=["total_y_definition"]
        )

        # Lower bound
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=self.order_name,
                                       val=x_obj )],
            senses=["G"],
            rhs=[self.wave_size_lb],
            names=["total_y_definition"]
        )
        
        # Upper bound
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=self.order_name,
                                       val=x_obj )],
            senses=["L"],
            rhs=[self.wave_size_ub],
            names=["total_y_definition"]
        )
        
        for k in self.items:
            lhs_vars = []
            lhs_coefs = []
            for i, order in enumerate(self.orders):
                qty = order.get(k, 0)
                if qty > 0:
                    lhs_vars.append(self.order_name[i])
                    lhs_coefs.append(qty)
    
            rhs_vars = []
            rhs_coefs = []
            for j, aisle in enumerate(self.aisles):
                qty = aisle.get(k, 0)
                if qty > 0:
                    rhs_vars.append(self.aisle_name[j])
                    rhs_coefs.append(qty)
    
            # sum_i x_i * a_ik - sum_j y_j * b_jk <= 0
            vars_ = lhs_vars + rhs_vars
            coefs_ = lhs_coefs + [-c for c in rhs_coefs]
    
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=vars_, val=coefs_)],
                senses=["L"],
                rhs=[0],
                names=[f"item_{k}_constraint"]
            )
            
        # Will be replaced 
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[], val=[])],
            senses=["L"],
            rhs=[0],
            names=["n_open_aisles_constraint"]
        )
        
        self.model = model 
        
       
    def set_warm_start(self, initial_solution : dict = None):
        initial_solution = initial_solution or self.last_solution
        if not initial_solution:
            return 
        model = self.model
        
        var_names = self.last_solution["selected_orders"] + self.last_solution["selected_aisles"]
        solution_values = len(var_names) * [1]
        model.MIP_starts.add(
            cplex.SparsePair(ind=var_names, val=solution_values),
            self.model.MIP_starts.effort_level.auto
        )
         
        
    def solve_model(self, time_limit : int = None):
        
        model =self.model
        if self.warm_start:
            self.set_warm_start()
        
        if time_limit:
            model.parameters.timelimit.set(time_limit)
            
        model.solve()

        print("Solution status:", model.solution.get_status_string())
        
        if not model.solution.is_primal_feasible():
            return 0
        
        print("Objective value:", model.solution.get_objective_value())
        selected_orders = [i for i in self.order_name if model.solution.get_values(i) > 0.9]
        selected_aisles = [j for j in self.aisle_name if model.solution.get_values(j) > 0.9]
        print("Selected orders:", selected_orders)
        print("Selected aisles:", selected_aisles)
        self.last_solution = {"selected_orders" :selected_orders, 
                              "selected_aisles": selected_aisles}
        
        return model.solution.get_objective_value()
        
    def redefine_n_aisle_constraint(self, n_max):
        model = self.model
        
        model.linear_constraints.delete("n_open_aisles_constraint")
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["n_open_aisles"], val=[1])],
            senses=["L"],
            rhs=[n_max],
            names=["n_open_aisles_constraint"]
        )

    def solve(self, start):
        start_time = time.time()
        end_time = start_time + self.time_limit
        self.write_model()
        min_aisle = self.define_first_aisle(self.time_limit)
        best_ratio = 0
        for i in range(max(1, int(min_aisle)), self.n_aisles + 1):
            self.redefine_n_aisle_constraint(i)
            intermediate_time = time.time()
            FO = self.solve_model(time_limit = end_time - intermediate_time)
            
            logging.info(f"-- N aisle = {i} --")
            logging.info(f"Time Spent : {time.time() - intermediate_time}")
            if FO/ i >= best_ratio:
                logging.info(f" ------------------------------------ new best")
                best_ratio = FO/i
                self.best_sol = {"N_aisle" : i,
                                 "FO" : FO/i,
                                 "solution" : self.last_solution}
                
            if end_time <= time.time():
                logging.info(" ----------------------------- Interrupted Execution")
                break 
            if FO >= self.wave_size_ub:
                logging.info(" ----------------------------- Reached Upper Bound ")
                break
                
            logging.info(f"Ratio found : {FO/i} ")
            logging.info(f"Sum items : {FO} ")
            
        logging.info(f"\nBest ratio : {self.best_sol['FO']}")
        logging.info(f"N_aisle : {self.best_sol['N_aisle']} ")
        logging.info(f"Total Execution time : {time.time() - start_time} \n\n\n")

    def define_first_aisle(self, time_limit):
        self.write_model()
        model = self.model        
        #Min aisles
        new_FO = [(i, 1) for i in self.aisle_name]
        new_FO +=  [(i, 0) for i in self.order_name]
        self.model.objective.set_sense(model.objective.sense.minimize)
        self.model.objective.set_linear(new_FO)
        
        start_time = time.time() 
        min_aisle = self.solve_model(time_limit/3)
        logging.info(f"---------- Min aisle found : {min_aisle}\nTime Wasted : {time.time()- start_time}")
        # Redefine FO
        p_j = list(self.p_j.values())
        old_FO = [(i, p_j[j]) for j,i in enumerate(self.order_name)]
        model.objective.set_linear(old_FO)
        model.objective.set_sense(model.objective.sense.maximize)
        
        return min_aisle
        