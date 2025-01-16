# ChallengeSBPO2025

Welcome to Mercado Libre's 2025 SBPO Challenge repository! Here you will find the framework base code, documentation, and other resources related to the challenge.

## Change Log

- **15-01-2025**: Base framework code, documentation and dataset `A`.

## Project Structure

- `src/main/java/org/sbpo2025/challenge/Challenge.java`: Main Java class for reading an input, solving the challenge, and writing the output.
- `src/main/java/org/sbpo2025/challenge/ChallengeSolver.java`: Java class responsible for solving the wave order picking problem. Most of the solving logic should be implemented here.
- `src/main/java/org/sbpo2025/challenge/ChallengeSolution.java`: Java class representing the solution to the wave order picking problem.
- `src/main/resources/instances/`: Directory containing input instance files.
- `run_challenge.py`: Python script to compile code, run benchmarks, and evaluate solutions.
- `checker.py`: Python script for evaluating the feasibility and objective value of solutions.
- `docs/problem_description_en.pdf`: Document describing the problem in detail.
- `docs/challenge_rules_en.pdf`: Document outlining the rules of the challenge.

## Prerequisites

- Java 11
- Maven
- Python 3.8 or higher
- CPLEX 22.11 (optional)
- OR-Tools 4.11 (optional)

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/mercadolibre/challenge-sbpo-2025
    cd challenge-sbpo-2025
    ```
2. Set the paths to CPLEX and OR-Tools libraries in `run_challenge.py` if needed, e.g.:
    ```sh
    cplex_path = "$HOME/CPLEX_Studio2211/opl/bin/arm64_osx/"
    or_tools_path = "$HOME/Documents/or-tools/build/lib/"
    ```

## Usage

### Running the challenge

To compile the code and run benchmarks, use the following command:
```sh
python run_challenge.py <source_folder> <input_folder> <output_folder>
```
Where `<source_folder>` is the path to the Java source code, more specifically, where the `pom.xml` file is located.

In order to run this script you will need the `timeout` (or `gtimeout` on macOS) command installed. You can install it using `apt-get install coreutils` (or equivalent) on Linux or `brew install coreutils` on macOS.

### Checking solution viability

To check the feasibility and objective value of a solution, use the following command:
```sh
python checker.py <input_file> <solution_file>
```

## Examples

1. Compile and run benchmarks:
    ```sh
    python run_challenge.py src/main/java/org/sbpo2025/challenge src/main/resources/instances output
    ```
   
2. Check solution viability:
    ```sh
    python checker.py src/main/resources/instances/instance_001.txt output/instance_001.txt
    ```
