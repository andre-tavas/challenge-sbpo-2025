package org.sbpo2025.challenge;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class BinaryGeneticAlgorithm {
    private final int populationSize;
    private final int chromosomeLength;
    private final double crossoverRate;
    private final double mutationRate;
    private final int elitismCount;
    private final int tournamentSize;
    private final int maxGenerations;
    private final Random random;
    private final Function<Set<Integer>, Double> fitnessFunction;

    private List<Individual> population;
    private int generation;
    private Individual bestIndividual;

    public BinaryGeneticAlgorithm(
            int populationSize,
            int chromosomeLength,
            double crossoverRate,
            double mutationRate,
            int elitismCount,
            int tournamentSize,
            int maxGenerations,
            Function<Set<Integer>, Double> fitnessFunction) {
        this.populationSize = populationSize;
        this.chromosomeLength = chromosomeLength;
        this.crossoverRate = crossoverRate;
        this.mutationRate = mutationRate;
        this.elitismCount = elitismCount;
        this.tournamentSize = tournamentSize;
        this.maxGenerations = maxGenerations;
        this.fitnessFunction = fitnessFunction;
        this.random = new Random();
        this.generation = 0;
    }

    public void initialize() {
        population = new ArrayList<>(populationSize);
        for (int i = 0; i < populationSize; i++) {
            population.add(new Individual(chromosomeLength, random, fitnessFunction));
        }
        evaluatePopulation();
        updateBestIndividual();
    }

    public void evaluatePopulation() {
        population.forEach(Individual::calculateFitness);
        population.sort(Comparator.comparingDouble(Individual::getFitness).reversed());
    }

    public void updateBestIndividual() {
        Individual currentBest = population.get(0);
        if (bestIndividual == null || currentBest.getFitness() > bestIndividual.getFitness()) {
            bestIndividual = new Individual(currentBest);
        }
    }

    public void evolve() {
        List<Individual> newPopulation = new ArrayList<>(populationSize);

        // Elitism: Keep the best individuals
        for (int i = 0; i < elitismCount; i++) {
            newPopulation.add(new Individual(population.get(i)));
        }

        // Fill the rest of the population with crossover and mutation
        while (newPopulation.size() < populationSize) {
            Individual parent1 = selectParent();
            Individual parent2 = selectParent();
            
            if (random.nextDouble() < crossoverRate) {
                Individual[] children = crossover(parent1, parent2);
                for (Individual child : children) {
                    if (newPopulation.size() < populationSize) {
                        mutate(child);
                        child.calculateFitness();
                        newPopulation.add(child);
                    }
                }
            } else {
                if (newPopulation.size() < populationSize) {
                    Individual child1 = new Individual(parent1);
                    mutate(child1);
                    child1.calculateFitness();
                    newPopulation.add(child1);
                }
                
                if (newPopulation.size() < populationSize) {
                    Individual child2 = new Individual(parent2);
                    mutate(child2);
                    child2.calculateFitness();
                    newPopulation.add(child2);
                }
            }
        }

        population = newPopulation;
        generation++;
        evaluatePopulation();
        updateBestIndividual();
    }

    public Individual selectParent() {
        // Tournament selection
        List<Individual> tournament = new ArrayList<>(tournamentSize);
        for (int i = 0; i < tournamentSize; i++) {
            int randomIndex = random.nextInt(populationSize);
            tournament.add(population.get(randomIndex));
        }
        return tournament.stream()
                .max(Comparator.comparingDouble(Individual::getFitness))
                .orElse(population.get(0));
    }

    public Individual[] crossover(Individual parent1, Individual parent2) {
        // Single-point crossover
        Individual child1 = new Individual(chromosomeLength, random, fitnessFunction);
        Individual child2 = new Individual(chromosomeLength, random, fitnessFunction);
        
        int crossoverPoint = random.nextInt(chromosomeLength);
        
        for (int i = 0; i < chromosomeLength; i++) {
            if (i < crossoverPoint) {
                child1.setGene(i, parent1.getGene(i));
                child2.setGene(i, parent2.getGene(i));
            } else {
                child1.setGene(i, parent2.getGene(i));
                child2.setGene(i, parent1.getGene(i));
            }
        }
        
        return new Individual[]{child1, child2};
    }

    public void mutate(Individual individual) {
        for (int i = 0; i < chromosomeLength; i++) {
            if (random.nextDouble() < mutationRate) {
                individual.flipGene(i);
            }
        }
    }

    public void run() {
        initialize();
        
        System.out.println("Starting Genetic Algorithm");
        System.out.println("Generation 0 | Best Fitness: " + bestIndividual.getFitness());
        
        for (int i = 0; i < maxGenerations; i++) {
            evolve();
            
            if (i % 10 == 0 || i == maxGenerations - 1) {
                System.out.println("Generation " + generation + 
                                  " | Best Fitness: " + bestIndividual.getFitness() + 
                                  " | Best Solution: " + bestIndividual.positionsAsString());
            }
        }
        
        System.out.println("\nFinal result after " + generation + " generations:");
        System.out.println("Best Fitness: " + bestIndividual.getFitness());
        System.out.println("Chromosome: " + bestIndividual);
        System.out.println("Positions of 1s: " + bestIndividual.positionsAsString());
    }

    public Individual getBestIndividual() {
        return bestIndividual;
    }

    public static class Individual {
        private boolean[] chromosome;
        private double fitness;
        private final Random random;
        private final Function<Set<Integer>, Double> fitnessFunction;

        public Individual(int length, Random random, Function<Set<Integer>, Double> fitnessFunction) {
            this.chromosome = new boolean[length];
            this.random = random;
            this.fitnessFunction = fitnessFunction;
            
            // Initialize with random genes
            for (int i = 0; i < length; i++) {
                chromosome[i] = random.nextBoolean();
            }
        }

        public Individual(Individual other) {
            this.chromosome = Arrays.copyOf(other.chromosome, other.chromosome.length);
            this.fitness = other.fitness;
            this.random = other.random;
            this.fitnessFunction = other.fitnessFunction;
        }

        public boolean getGene(int index) {
            return chromosome[index];
        }

        public void setGene(int index, boolean value) {
            this.chromosome[index] = value;
        }

        public void flipGene(int index) {
            this.chromosome[index] = !this.chromosome[index];
        }

        public int getLength() {
            return chromosome.length;
        }

        public double getFitness() {
            return fitness;
        }

        public void calculateFitness() {
            // Convert the binary chromosome to a set of positions where genes are 1
            Set<Integer> positions = new HashSet<>();
            for (int i = 0; i < chromosome.length; i++) {
                if (chromosome[i]) {
                    positions.add(i);
                }
            }
            
            // Use the provided fitness function to evaluate
            this.fitness = fitnessFunction.apply(positions);
        }

        public Set<Integer> getPositionsOfOnes() {
            Set<Integer> positions = new HashSet<>();
            for (int i = 0; i < chromosome.length; i++) {
                if (chromosome[i]) {
                    positions.add(i);
                }
            }
            return positions;
        }
        
        public String positionsAsString() {
            return getPositionsOfOnes().toString();
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (boolean gene : chromosome) {
                sb.append(gene ? "1" : "0");
            }
            return sb.toString();
        }
    }

    public static void main(String[] args) {
        // Example fitness function - maximize the sum of positions
        // (This is just a simple example, replace with your actual fitness function)
        Function<Set<Integer>, Double> exampleFitnessFunction = positions -> {
            return (double) positions.stream().mapToInt(Integer::intValue).sum();
        };
        
        BinaryGeneticAlgorithm ga = new BinaryGeneticAlgorithm(
            100,           // populationSize
            20,            // chromosomeLength
            0.7,           // crossoverRate
            0.01,          // mutationRate
            5,             // elitismCount
            5,             // tournamentSize
            100,           // maxGenerations
            exampleFitnessFunction
        );
        
        ga.run();
    }
}