import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

import random
import time
import numpy as np

class Brkga:
    def __init__(self,
                 *,
                 problem_object,
                 population_size: int = 10,
                 elite_size: int = 0.10,
                 mutation_size: int = 0.20,
                 inheritance_p: float = 0.5,
                 time_limit: int = 300,
                 ):
        
    
        for name, value in problem_object.__dict__.items():
            setattr(self, name, value)
        for name in dir(problem_object):
            if callable(getattr(problem_object, name)) and not name.startswith("__"):
                setattr(self, name, getattr(problem_object, name))
            
        self.number_allele = len(self.allele_names)
        
        self.population_size = self.number_allele * population_size
        self.elite_size = int(self.population_size * elite_size) 
        self.mutation_size = int(self.population_size * mutation_size) 
        
        self.new_generation_size = self.population_size - self.elite_size - self.mutation_size
        self.inheritance_p = inheritance_p
        
        self.time_limit = None
        self.it_without_improve = None
        self.best = None
        self.FO = None
        self.solution = None
        self.it = 0
        
        # Generate T0 population
        self.population = self.generate_random(self.population_size)
        
        self.population_info_dict = self.avaliate_population(self.population)
        
    @abstractmethod
    def evaluate_cromossome(self):
        pass
        
    def generate_random(self, size):
        return [self.generate_cromossome() for i in range(size)]  
    
    def generate_cromossome(self):
        new_cromossome = list(np.random.random(self.number_allele))        
        return new_cromossome 
    
    def cross_over(self, ind_elite, ind_non_elite):
        new_ind = ind_elite.copy()
        for i in range(len(new_ind)):
            prob = random.random()
            if prob >= self.inheritance_p:
                new_ind[i] = ind_non_elite[i] 
        return new_ind
      
    def get_elite_population(self):
        ID_fitness_ranking = sorted(self.population_info_dict.items(), key=lambda item: item[1]["Fitness_Value"], reverse= True)
        if self.solution is None or self.FO < ID_fitness_ranking[0][1]['Fitness_Value']:            
            self.FO = ID_fitness_ranking[0][1]['Fitness_Value']
            self.solution = ID_fitness_ranking[0][1]['Solution']
            self.best = ID_fitness_ranking[0][1]['Random_Keys']
            self.best_ind = ID_fitness_ranking[0][0]
            self.it_without_improve = 0
        else:
            self.it_without_improve += 1
        
        elite_info_dict = dict(ID_fitness_ranking[:self.elite_size])
        non_elite_info_dict = dict(ID_fitness_ranking[self.elite_size:] )
    
        return elite_info_dict, non_elite_info_dict
            
    def avaliate_population(self, population : list[float]):
        
        population_info_dict = {}
        for i, ind in enumerate(population):
            sol, fitness = self.evaluate_cromossome(ind)
            population_info_dict[(i, self.it)] = {
                "Random_Keys": ind,
                "Fitness_Value": fitness,
                "Solution": sol
            }
        return population_info_dict
    
    def create_new_population(self, elite_info_dict, non_elite_info_dict):
        
        mutants = self.generate_random(self.mutation_size)     
        
        new_population = []
        for _ in range(self.new_generation_size):
            # Choosing randomly two cromossomes
            ind_non_elite = random.choice(list(non_elite_info_dict.keys()))
            ind_elite = random.choice(list(elite_info_dict.keys()))     
            # Crossing over
            new_ind = self.cross_over(ind_elite = elite_info_dict[ind_elite]['Random_Keys'],
                                      ind_non_elite = non_elite_info_dict[ind_non_elite]['Random_Keys'])           
            new_population.append(new_ind)
            
        new_population += mutants
        new_population_info_dict = self.avaliate_population(new_population)  
        new_population_info_dict.update(elite_info_dict)
        
        elite_alleles = [elite_info_dict[ID]['Random_Keys'] for ID in elite_info_dict]
                
        self.population = new_population + elite_alleles
        self.population_info_dict = new_population_info_dict
                
    def run_algorithm(self,
                      time_limit: int = 180,
                      it_max: int = 100000,
                      it_max_whithout_improve: int = 10000
                      ):
        
        self.time_limit = time_limit 
        self.it_without_improve = 0
        it_completed = 0 
        start_time = time.time()
        history_best = []  
        time_spend = time.time() - start_time  
        while self.stop_condition(time_spend, it_max_whithout_improve, it_completed, it_max):
                  
            elite_info_dict, non_elite_info_dict = self.get_elite_population()             
            it_completed += 1
            self.it += 1
            self.create_new_population(elite_info_dict, non_elite_info_dict)
            history_best.append(self.FO)
            time_spend = time.time() - start_time
            
        #Plot Graph    
        IT = list(range(it_completed))
        plt.plot(IT, history_best)
        
        upper_y = history_best[-1]
        lower_y = upper_y * 2
        plt.ylim(lower_y, upper_y)

        plt.title('IT vs Fitness')
        plt.xlabel('IT')
        plt.ylabel('Fitnaess Value')
        plt.grid(True)
        plt.show()
        
    def stop_condition(self, time_spend, it_max_whithout_improve, it_completed, it_max):
        if time_spend > self.time_limit:
            print("Tempo limite foi alcançado")
            return False
        if it_max_whithout_improve < self.it_without_improve:
            print("Máximo de iterações sem melhoria foi alcançada")
            return False
        if it_completed > it_max:
            print("Número máximo de iterações foi alcançada")
            return False
        return True
        
        
                


   