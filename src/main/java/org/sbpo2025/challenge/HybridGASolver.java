package org.sbpo2025.challenge;

import ilog.concert.*;
import ilog.cplex.*;
import java.util.*;
import org.apache.commons.lang3.time.StopWatch;

public class HybridGASolver extends ChallengeSolver {

    protected int[] totalItemsPerOrder;

    public HybridGASolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB,
            int waveSizeUB) {
        super(orders, aisles, nItems, waveSizeLB, waveSizeUB);
        this.totalItemsPerOrder = computeTotalItemsPerOrder(orders);
    }

    public static int[] computeTotalItemsPerOrder(List<Map<Integer, Integer>> listOfMappings) {
        int[] totalItemsPerOrder = new int[listOfMappings.size()];
        for (int i = 0; i < listOfMappings.size(); i++) {
            for (int quantity : listOfMappings.get(i).values()) {
                totalItemsPerOrder[i] += quantity;
            }
        }
        return totalItemsPerOrder;
    }

    public static class Individual {
        public Set<Integer> aisles = new HashSet<>();
        public double fitness = 0.0;
        public Set<Integer> selectedOrders = new HashSet<>();
        public boolean feasible = false;

        public Individual() {
        }

        public Individual(Set<Integer> aisles, double fitness, Set<Integer> selectedOrders, boolean feasible) {
            this.aisles = aisles;
            this.fitness = fitness;
            this.selectedOrders = selectedOrders;
            this.feasible = feasible;
        }
    }

    public Individual evaluateIndividual(Set<Integer> selectedAisles) {
        try {

            int[] totalUnitsAvailable = computeTotalUnitsAvailable(selectedAisles);

            IloCplex cplex = new IloCplex();
            cplex.setParam(IloCplex.Param.RandomSeed, RandomSeed);
            cplex.setWarning(null);
            cplex.setOut(null); // Suppress output

            IloNumVar[] x = cplex.boolVarArray(orders.size());

            // Objective: Maximize total picked items
            IloLinearNumExpr objective = cplex.scalProd(totalItemsPerOrder, x);
            cplex.addMaximize(objective);

            // Constraint: item availability
            for (int i = 0; i < nItems; i++) {
                IloLinearNumExpr totalItem_i = cplex.linearNumExpr();
                for (int o = 0; o < orders.size(); o++) {
                    if (orders.get(o).containsKey(i)) {
                        totalItem_i.addTerm(orders.get(o).get(i), x[o]);
                    }
                }
                cplex.addLe(totalItem_i, totalUnitsAvailable[i]);
            }

            // Solve
            if (!cplex.solve()) {
                cplex.end();
                return new Individual(); // infeasible
            }

            double nbItemsCollected = cplex.getObjValue();
            Set<Integer> selectedOrders = new HashSet<>();
            for (int o = 0; o < orders.size(); o++) {
                if (cplex.getValue(x[o]) > 0.9) {
                    selectedOrders.add(o);
                }
            }

            cplex.end();

            double fitness = nbItemsCollected / aisles.size();
            return new Individual(selectedAisles, fitness, selectedOrders, true);

        } catch (IloException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static Individual generateRandomIndividual(int nbAisles, Random rng) {
        // GENERATE A 
        int size = rng.nextInt(aisles.size()) + 1;
        Set<Integer> aisles = new HashSet<>();
        while (aisles.size() < size) {
            aisles.add(rng.nextInt(nbAisles));
        }
        return evaluateIndividual(input, aisles);
    }
}

//Individual class
class Individual {

    public Set<Integer> visitedAisles = new HashSet<>();
    public int[] genes;
    public double fitness = 0.0;
    public Set<Integer> selectedOrders = new HashSet<>();
    public boolean feasible = false;

    public Individual() {
        Random rn = new Random();

        //Set genes randomly for each individual
        for (int i = 0; i < genes.length; i++) {
            genes[i] = Math.abs(rn.nextInt() % 2);
        }

        fitness = 0;
    }

    //Calculate fitness
    public void calcFitness() {

        fitness = 0;
        for (int i = 0; i < 5; i++) {
            if (genes[i] == 1) {
                ++fitness;
            }
        }
    }

}

//Population class
class Population {

    int popSize = 10;
    Individual[] individuals = new Individual[10];
    int fittest = 0;

    //Initialize population
    public void initializePopulation(int size) {
        for (int i = 0; i < individuals.length; i++) {
            individuals[i] = new Individual();
        }
    }

    //Get the fittest individual
    public Individual getFittest() {
        int maxFit = Integer.MIN_VALUE;
        int maxFitIndex = 0;
        for (int i = 0; i < individuals.length; i++) {
            if (maxFit <= individuals[i].fitness) {
                maxFit = individuals[i].fitness;
                maxFitIndex = i;
            }
        }
        fittest = individuals[maxFitIndex].fitness;
        return individuals[maxFitIndex];
    }

    //Get the second most fittest individual
    public Individual getSecondFittest() {
        int maxFit1 = 0;
        int maxFit2 = 0;
        for (int i = 0; i < individuals.length; i++) {
            if (individuals[i].fitness > individuals[maxFit1].fitness) {
                maxFit2 = maxFit1;
                maxFit1 = i;
            } else if (individuals[i].fitness > individuals[maxFit2].fitness) {
                maxFit2 = i;
            }
        }
        return individuals[maxFit2];
    }

    //Get index of least fittest individual
    public int getLeastFittestIndex() {
        int minFitVal = Integer.MAX_VALUE;
        int minFitIndex = 0;
        for (int i = 0; i < individuals.length; i++) {
            if (minFitVal >= individuals[i].fitness) {
                minFitVal = individuals[i].fitness;
                minFitIndex = i;
            }
        }
        return minFitIndex;
    }

    //Calculate fitness of each individual
    public void calculateFitness() {

        for (int i = 0; i < individuals.length; i++) {
            individuals[i].calcFitness();
        }
        getFittest();
    }

}