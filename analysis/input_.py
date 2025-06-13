import time
from typing import List, Dict
import logging
import os

class Challenge:
    def __init__(self):
        self.orders: List[Dict[int, int]] = []
        self.aisles: List[Dict[int, int]] = []
        self.n_items: int = 0
        self.wave_size_lb: int = 0
        self.wave_size_ub: int = 0

    def read_input(self, input_file_path: str):
        try:
            with open(input_file_path, 'r') as f:
                first_line = f.readline().strip().split()
                n_orders, self.n_items, n_aisles = map(int, first_line)

                # Read orders
                self.orders = self._read_item_quantity_pairs(f, n_orders)

                # Read aisles
                self.aisles = self._read_item_quantity_pairs(f, n_aisles)

                # Read wave size bounds
                bounds = f.readline().strip().split()
                self.wave_size_lb, self.wave_size_ub = map(int, bounds)
        except IOError as e:
            print(f"Error reading input from {input_file_path}")
            raise e


    def _read_item_quantity_pairs(self, file_obj, n_lines: int) -> List[Dict[int, int]]:
        result = []
        for _ in range(n_lines):
            parts = file_obj.readline().strip().split()
            n_items = int(parts[0])
            entry = {
                int(parts[2 * k + 1]): int(parts[2 * k + 2])
                for k in range(n_items)
            }
            result.append(entry)
        return result

    def write_output(self, challenge_solution, output_file_path: str):
        if challenge_solution is None:
            print("Solution not found")
            return
        try:
            with open(output_file_path, 'w') as f:
                orders = challenge_solution.orders
                aisles = challenge_solution.aisles

                f.write(f"{len(orders)}\n")
                for order in orders:
                    f.write(f"{order}\n")

                f.write(f"{len(aisles)}\n")
                for aisle in aisles:
                    f.write(f"{aisle}\n")
            print(f"Output written to {output_file_path}")
        except IOError as e:
            print(f"Error writing output to {output_file_path}")
            raise e


class ChallengeSolution:
    def __init__(self, orders: List[int], aisles: List[int]):
        self.orders = orders
        self.aisles = aisles

 
class ChallengeSolver:
    def __init__(self, orders, aisles, n_items, wave_lb, wave_ub):
        self.orders = orders
        self.aisles = aisles
        self.n_items = n_items
        self.items = list(range(n_items))
        self.wave_size_lb = wave_lb
        self.wave_size_ub = wave_ub


    def solve(self, start_time) -> ChallengeSolution:
        # TODO: Implement your logic here
        # For now, a dummy solution:
        orders = list(range(min(3, len(self.orders))))
        aisles = list(range(min(2, len(self.aisles))))
        elapsed_time = time.perf_counter() - start_time
        print(f"Solved in {elapsed_time:.3f} seconds")
        return ChallengeSolution(orders, aisles)



#Run all instances and create log analysis
if __name__ == "__main__":
    import sys
    from Approche1 import Approche1

    logging.basicConfig(
        filename='Approach2.log',
        level=logging.INFO,
        format='%(message)s',
    )

    instance_types_path = ["../datasets/a", "../datasets/b" ]

    
    for path in instance_types_path:
        logging.info(f" ----- Run instances type: {path[-1]} ---------\n")
        for instance in os.listdir(path):

            logging.info(f" -------- {instance} --------")
            input_file = os.path.join(path, instance)

            start = time.perf_counter()
        
            challenge = Challenge()
            challenge.read_input(input_file)

            solver = Approche1(challenge.orders, challenge.aisles, challenge.n_items,
                                     challenge.wave_size_lb, challenge.wave_size_ub)
            solution = solver.solve(start)
            
        logging.info("\n\n\n\n")
        
## Solve specific instance
# if __name__ == "__main__":
#     import sys
#     from Approche1 import Approche1

#     logging.basicConfig(
#         filename='Approach2.log',
#         level=logging.INFO,
#         format='%(message)s',
#     )

#     instance_types_path = ["../datasets/a", "../datasets/b" ]

    

#     path = "../datasets/a"
#     instance = "instance_0010.txt"
#     logging.info(f" -------- {instance} --------")
#     input_file = os.path.join(path, instance)

#     start = time.perf_counter()

#     challenge = Challenge()
#     challenge.read_input(input_file)

#     solver = Approche1(challenge.orders, challenge.aisles, challenge.n_items,
#                              challenge.wave_size_lb, challenge.wave_size_ub)
#     solution = solver.solve(start)
            
#     logging.info("\n\n\n\n")


# if __name__ == "__main__":
#     import sys
#     from Approche1 import Approche1

#     # logging.basicConfig(
#     #     filename='Approach1.log',
#     #     level=logging.INFO,
#     #     format='%(message)s',
#     # )

#     # if len(sys.argv) != 3:
#     #     print("Usage: python challenge.py <inputFilePath> <outputFilePath>")
#     #     sys.exit(1)

#     # input_file = sys.argv[1]
#     # output_file = sys.argv[2]
    
#     input_file = "../datasets/a/instance_0002.txt"
#     output_file = "output.txt"

#     start = time.perf_counter()

#     challenge = Challenge()
#     challenge.read_input(input_file)

#     solver = Approche1(challenge.orders, challenge.aisles, challenge.n_items,
#                              challenge.wave_size_lb, challenge.wave_size_ub)

#     solution = solver.solve(start)
#     challenge.write_output(solution, output_file)
