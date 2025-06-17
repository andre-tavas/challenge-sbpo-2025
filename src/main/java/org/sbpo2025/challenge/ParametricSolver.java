package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;
import ilog.concert.*;
import ilog.cplex.*;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class ParametricSolver extends ChallengeSolver {

    private static final double EPS = 1e-4;
    private static final double TIME_MARGIN = 20.0;
    private static final double TIME_LIMIT_SUBPROBLEM = 60.0;
    private static final int LARGE_INSTANCE = 30000; // considerando n pedidso + items

    // Precomputed data structures for efficiency
    private int[] orderTotals;
    private Map<Integer, List<Integer>> itemToOrders;
    private Map<Integer, List<Integer>> itemToAisles;

    public ParametricSolver(List<Map<Integer, Integer>> orders,
                            List<Map<Integer, Integer>> aisles,
                            int nItems, int waveSizeLB, int waveSizeUB) {
        super(orders, aisles, nItems, waveSizeLB, waveSizeUB);
        System.out.println("\n==================================================================================");
        precomputeDataStructures();
    }

    private void precomputeDataStructures() {
        long start = System.currentTimeMillis();
        
        // Precompute order totals
        orderTotals = new int[orders.size()];
        for (int o = 0; o < orders.size(); o++) {
            orderTotals[o] = orders.get(o).values().stream().mapToInt(i -> i).sum();
        }

        // Create efficient lookup maps
        itemToOrders = new HashMap<>();
        itemToAisles = new HashMap<>();

        for (int i = 0; i < nItems; i++) {
            itemToOrders.put(i, new ArrayList<>());
            itemToAisles.put(i, new ArrayList<>());
        }

        // Populate item-to-orders mapping
        for (int o = 0; o < orders.size(); o++) {
            for (int item : orders.get(o).keySet()) {
                itemToOrders.get(item).add(o);
            }
        }

        // Populate item-to-aisles mapping
        for (int a = 0; a < aisles.size(); a++) {
            for (int item : aisles.get(a).keySet()) {
                itemToAisles.get(item).add(a);
            }
        }

        long end = System.currentTimeMillis();
        System.out.printf("Data preprocessing completed in %d ms%n", end - start);
    }

    @Override
    public ChallengeSolution solve(StopWatch stopWatch) {
        long totalStartTime = System.currentTimeMillis();
        boolean hasConverged = false;
        double finalObjective = 0.0;
        
        try {
            long modelBuildStart = System.currentTimeMillis();
            
            double bestRatio = 0.0;
            ChallengeSolution bestSol = null;
            int nO = orders.size(), nA = aisles.size();

            IloCplex cplex = new IloCplex();
            
            // Optimize CPLEX settings for performance
            configureCplex(cplex, stopWatch);

            // Create variables
            IloIntVar[] x = new IloIntVar[nO];
            IloIntVar[] y = new IloIntVar[nA];
            
            for (int o = 0; o < nO; o++) {
                x[o] = cplex.boolVar();  // Remove names for better performance
            }
            for (int a = 0; a < nA; a++) {
                y[a] = cplex.boolVar();
            }

            // Build expressions more efficiently
            IloLinearNumExpr totalItemsPerOrder = cplex.linearNumExpr();
            for (int o = 0; o < nO; o++) {
                if (orderTotals[o] > 0) {  // Skip empty orders
                    totalItemsPerOrder.addTerm(orderTotals[o], x[o]);
                }
            }

            // Wave size constraints
            cplex.addGe(totalItemsPerOrder, waveSizeLB);
            cplex.addLe(totalItemsPerOrder, waveSizeUB);

            // Efficient constraint generation using precomputed mappings
            addCoverageConstraints(cplex, x, y);

            // Expression for total aisles
            IloLinearNumExpr totalAislesExpr = cplex.linearNumExpr();
            for (int a = 0; a < nA; a++) {
                totalAislesExpr.addTerm(1.0, y[a]);
            }

            long modelBuildEnd = System.currentTimeMillis();
            System.out.printf("Model building completed in %d ms (%.3f seconds)%n", 
                            modelBuildEnd - modelBuildStart, 
                            (modelBuildEnd - modelBuildStart) / 1000.0);

            // Use objective modification instead of recreation
            IloObjective obj = cplex.addMaximize(totalItemsPerOrder);
            
            double t = 0;
            List<Integer> bestO = null;
            List<Integer> bestA = null;
            int iteration = 0;
            int maxIterations = Math.min(100, nO + nA); // Limit iterations for large instances

            // Initialize the gap
            double relativeGap = 0.0;

            while (getRemainingTime(stopWatch) > TIME_MARGIN && iteration < maxIterations) {
                long iterStart = System.currentTimeMillis();
                
                System.out.printf("Iteration %d: t = %.3f, Remaining Time = %d%n",
                    iteration, t, getRemainingTime(stopWatch));

                // Modify objective instead of recreating
                IloNumExpr newObj = cplex.sum(totalItemsPerOrder, cplex.prod(-t, totalAislesExpr));
                obj.setExpr(newObj);
                
                // Se gap da ultima iteracao foi mt alto, usar todo o tempo restante no time limit
                double time_limit = TIME_LIMIT_SUBPROBLEM;
                if(relativeGap > 1){
                    System.out.print("Using all remaning time for the subproblem.\n");
                    time_limit = (double)getRemainingTime(stopWatch);
                }
                
                cplex.setParam(IloCplex.Param.TimeLimit, 
                              Math.min(getRemainingTime(stopWatch) - TIME_MARGIN, time_limit)); // Cap individual solve time

                if (!cplex.solve()) {
                    System.out.println("No feasible solution found in iteration " + iteration);
                    break;
                }

                try {
                    relativeGap = cplex.getMIPRelativeGap();
                } catch (IloException e) {
                    // If gap is not available, set to 0
                    relativeGap = 0.0;
                }

                // Extract solution more efficiently
                SolutionData solData = extractSolution(cplex, x, y);
                
                if (solData.aislesCount == 0) break;

                double newRatio = solData.totalItems / solData.aislesCount;
                
                long iterEnd = System.currentTimeMillis();
                System.out.printf("Iteration %d: Orders=%d, Aisles=%d, Ratio=%.3f, Gap=%.4f%%, Time=%dms%n",
                    iteration, solData.selectedOrders.size(), solData.selectedAisles.size(), 
                    newRatio, relativeGap * 100, iterEnd - iterStart);

                // Update best solution
                if (newRatio > bestRatio + EPS) {
                    bestRatio = newRatio;
                    bestO = new ArrayList<>(solData.selectedOrders);
                    bestA = new ArrayList<>(solData.selectedAisles);
                    System.out.printf("New best ratio: %.3f%n", bestRatio);
                }

                // Convergence check with adaptive tolerance
                double tolerance = Math.max(EPS, EPS * Math.abs(t));
                if (Math.abs(newRatio - t) < tolerance && relativeGap < 1) {
                    System.out.println("Converged at iteration " + iteration);
                    hasConverged = true;
                    break;
                }

                t = newRatio;
                iteration++;
            }

            cplex.end();

            // Set final objective value
            finalObjective = bestRatio;
            
            // Calculate total execution time and export results
            long totalEndTime = System.currentTimeMillis();
            double totalTimeSeconds = (totalEndTime - totalStartTime) / 1000.0;
            exportResults(totalTimeSeconds, finalObjective, hasConverged);

            if (bestO != null && bestA != null) {
                System.out.printf("Final solution: %d orders, %d aisles, ratio=%.3f%n", 
                                bestO.size(), bestA.size(), bestRatio);
                return new ChallengeSolution(new HashSet<>(bestO), new HashSet<>(bestA));
            } else {
                return randomFeasible();
            }

        } catch (IloException e) {
            e.printStackTrace();
            
            // Export results even if there was an exception
            long totalEndTime = System.currentTimeMillis();
            double totalTimeSeconds = (totalEndTime - totalStartTime) / 1000.0;
            exportResults(totalTimeSeconds, finalObjective, hasConverged);
            
            return null;
        }
    }

    private void configureCplex(IloCplex cplex, StopWatch stopWatch) throws IloException {
        // Output management
        try {
            PrintStream out = new PrintStream(new FileOutputStream("cplex_log.txt"));
            cplex.setOut(out);
            cplex.setWarning(out);
        } catch (FileNotFoundException e) {
            cplex.setOut(null);
            cplex.setWarning(null);
        }

        // Basic parameters
        cplex.setParam(IloCplex.Param.RandomSeed, RandomSeed);
        cplex.setParam(IloCplex.Param.TimeLimit, getRemainingTime(stopWatch));
        
        // Large instance specific parameters
        int instanceSize = orders.size() + nItems;
        
        if (instanceSize > LARGE_INSTANCE) {
            // LARGE INSTANCE SETTINGS
            
            // 1. Emphasis on finding feasible solutions quickly
            cplex.setParam(IloCplex.Param.Emphasis.MIP, IloCplex.MIPEmphasis.Feasibility);
            
            // 2. Aggressive preprocessing
            cplex.setParam(IloCplex.Param.Preprocessing.Presolve, true);
            cplex.setParam(IloCplex.Param.Preprocessing.NumPass, 10);
            cplex.setParam(IloCplex.Param.Preprocessing.Aggregator, 10);
            
            // 3. Relaxed tolerances for faster solving
            cplex.setParam(IloCplex.Param.MIP.Tolerances.MIPGap, 0.05); // 5% gap
            cplex.setParam(IloCplex.Param.MIP.Tolerances.AbsMIPGap, 0.5);
            cplex.setParam(IloCplex.Param.MIP.Tolerances.Integrality, 1e-4);
            
            // 4. Heuristic settings
            cplex.setParam(IloCplex.Param.MIP.Strategy.HeuristicFreq, 5); // More frequent
            cplex.setParam(IloCplex.Param.MIP.Strategy.RINSHeur, 50); // RINS every 50 nodes
            
            // 5. Node selection strategy
            cplex.setParam(IloCplex.Param.MIP.Strategy.NodeSelect, 1); // Best-bound
            
            // 6. Cut generation - be selective
            cplex.setParam(IloCplex.Param.MIP.Cuts.Cliques, 1); // Moderate
            cplex.setParam(IloCplex.Param.MIP.Cuts.Covers, 1);  // Moderate
            cplex.setParam(IloCplex.Param.MIP.Cuts.FlowCovers, 0); // Disable expensive cuts
            cplex.setParam(IloCplex.Param.MIP.Cuts.GUBCovers, 0);
            cplex.setParam(IloCplex.Param.MIP.Cuts.Implied, 1);
            cplex.setParam(IloCplex.Param.MIP.Cuts.MIRCut, 1);
            
            // 7. Branching strategy
            cplex.setParam(IloCplex.Param.MIP.Strategy.VariableSelect, 3); // Strong branching
            
            // 8. Memory and node limits
            cplex.setParam(IloCplex.Param.WorkMem, 4096); // 4GB working memory
            cplex.setParam(IloCplex.Param.MIP.Strategy.File, 2); // Disk storage
            cplex.setParam(IloCplex.Param.MIP.Limits.Nodes, 100000); // Limit nodes for time control

            // Threading - adaptive based on instance size
            int availableProcessors = Runtime.getRuntime().availableProcessors();
            int threads = instanceSize > 2000 ? Math.min(6, availableProcessors) : Math.min(4, availableProcessors);
            cplex.setParam(IloCplex.Param.Threads, threads);
            
            // Parallel mode for large instances
            if (instanceSize > 1500) {
                cplex.setParam(IloCplex.Param.Parallel, IloCplex.ParallelMode.Opportunistic);
            }
            
        } else {
            // MEDIUM INSTANCE SETTINGS
            // Aggressive performance settings
            cplex.setParam(IloCplex.Param.Emphasis.MIP, IloCplex.MIPEmphasis.Feasibility);
            cplex.setParam(IloCplex.Param.MIP.Strategy.HeuristicFreq, 10); // More frequent heuristics
            cplex.setParam(IloCplex.Param.Preprocessing.Presolve, true);
            cplex.setParam(IloCplex.Param.MIP.Cuts.Cliques, 2); // Aggressive clique cuts
            cplex.setParam(IloCplex.Param.MIP.Cuts.Covers, 2);  // Aggressive cover cuts
            
            // Threading
            int availableProcessors = Runtime.getRuntime().availableProcessors();
            cplex.setParam(IloCplex.Param.Threads, Math.min(4, availableProcessors));
            
            // Memory management
            cplex.setParam(IloCplex.Param.WorkMem, 2048); // 2GB working memory
            cplex.setParam(IloCplex.Param.MIP.Strategy.File, 2); // Node file on disk when needed
        }
    }

    private void addCoverageConstraints(IloCplex cplex, IloIntVar[] x, IloIntVar[] y) throws IloException {
        int constraintsAdded = 0;
        
        for (int i = 0; i < nItems; i++) {
            List<Integer> ordersWithItem = itemToOrders.get(i);
            List<Integer> aislesWithItem = itemToAisles.get(i);
            
            // Skip items not present in any order or aisle
            if (ordersWithItem.isEmpty() || aislesWithItem.isEmpty()) {
                continue;
            }

            IloLinearNumExpr lhs = cplex.linearNumExpr();
            for (int o : ordersWithItem) {
                lhs.addTerm(orders.get(o).get(i), x[o]);
            }

            IloLinearNumExpr rhs = cplex.linearNumExpr();
            for (int a : aislesWithItem) {
                rhs.addTerm(aisles.get(a).get(i), y[a]);
            }

            cplex.addLe(lhs, rhs);
            constraintsAdded++;
        }
        
        System.out.printf("Added %d coverage constraints%n", constraintsAdded);
    }

    private static class SolutionData {
        final List<Integer> selectedOrders;
        final List<Integer> selectedAisles;
        final double totalItems;
        final double aislesCount;

        SolutionData(List<Integer> orders, List<Integer> aisles, double items, double count) {
            this.selectedOrders = orders;
            this.selectedAisles = aisles;
            this.totalItems = items;
            this.aislesCount = count;
        }
    }

    private SolutionData extractSolution(IloCplex cplex, IloIntVar[] x, IloIntVar[] y) throws IloException {
        List<Integer> selO = new ArrayList<>();
        List<Integer> selA = new ArrayList<>();
        double items = 0.0;

        // Extract selected orders
        for (int o = 0; o < x.length; o++) {
            if (cplex.getValue(x[o]) > 0.5) {
                selO.add(o);
                items += orderTotals[o]; // Use precomputed totals
            }
        }

        // Extract selected aisles
        for (int a = 0; a < y.length; a++) {
            if (cplex.getValue(y[a]) > 0.5) {
                selA.add(a);
            }
        }

        return new SolutionData(selO, selA, items, selA.size());
    }

    private ChallengeSolution randomFeasible() {
        // Improved fallback: try to find a small feasible solution
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();
        
        // Find a minimal feasible solution
        int totalItems = 0;
        for (int o = 0; o < Math.min(orders.size(), 10) && totalItems < waveSizeLB; o++) {
            selectedOrders.add(o);
            totalItems += orderTotals[o];
            
            // Add required aisles for this order
            for (int item : orders.get(o).keySet()) {
                if (!itemToAisles.get(item).isEmpty()) {
                    selectedAisles.add(itemToAisles.get(item).get(0));
                }
            }
        }
        
        System.out.printf("Fallback solution: %d orders, %d aisles%n", 
                         selectedOrders.size(), selectedAisles.size());
        return new ChallengeSolution(selectedOrders, selectedAisles);
    }

    private void exportResults(double timeSeconds, double objective, boolean hasConverged) {
        try {
            java.io.File file = new java.io.File("parametric_results.csv");
            boolean fileExists = file.exists();
            
            // Use FileWriter with append=true to append to existing file
            try (PrintWriter writer = new PrintWriter(new java.io.FileWriter(file, true))) {
                // Write header only if file doesn't exist or is empty
                if (!fileExists || file.length() == 0) {
                    writer.println("time(s),objective,has_converged");
                }
                
                // Append data row
                writer.printf("%.3f,%.6f,%b%n", timeSeconds, objective, hasConverged);
                
                System.out.printf("Results appended to parametric_results.csv: time=%.3fs, objective=%.6f, converged=%b%n", 
                                timeSeconds, objective, hasConverged);
                
            }
        } catch (IOException e) {
            System.err.println("Error writing results to CSV file: " + e.getMessage());
            e.printStackTrace();
        }
    }

}