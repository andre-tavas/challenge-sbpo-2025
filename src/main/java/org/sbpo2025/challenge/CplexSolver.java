package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;
import ilog.concert.*;
import ilog.cplex.*;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class CplexSolver extends ChallengeSolver {

    public CplexSolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB, int waveSizeUB) {
        super(orders, aisles, nItems, waveSizeLB, waveSizeUB);
    }

    @Override
    public ChallengeSolution solve(StopWatch stopWatch) {
        try {
            //System.out.println("Solving with CPLEX using 2-stage iterative approach...");

            double bestRatio = 0.0;
            ChallengeSolution bestSolution = null;

            int maxAisles = aisles.size();
            double maxQuantity = 0;

            String[] auxOrdersVarName = new String[orders.size()];
            for (int i = 0; i < orders.size(); i++) {
                auxOrdersVarName[i] = "Order" + i + "Activation";
            }
            String[] auxAislesVarName = new String[aisles.size()];
            for (int i = 0; i < aisles.size(); i++) {
                auxAislesVarName[i] = "Aisle" + i + "Activation";;
            }

            IloCplex cplex = new IloCplex();
            cplex.setParam(IloCplex.Param.RandomSeed, RandomSeed);
            cplex.setOut(null);
            cplex.setWarning(null);
            Map<String, IloNumVar> quantity = createQuantityVariables(cplex);
            IloIntVar[] orderActivation = cplex.boolVarArray(orders.size(), auxOrdersVarName);
            IloIntVar[] aisleActivation = cplex.boolVarArray(aisles.size(), auxAislesVarName);
            addConstraints(cplex, quantity, orderActivation, aisleActivation);

            
            IloLinearNumExpr nbItemsCollected = cplex.linearNumExpr();
            for (int o = 0; o < orders.size(); o++) {
                for (int i : orders.get(o).keySet()) {
                    for (int a = 0; a < aisles.size(); a++) {
                        String key = "quantity_" + i + "_" + o + "_" + a;
                        if (quantity.containsKey(key)) {
                            nbItemsCollected.addTerm(1., quantity.get(key));
                        }
                    }
                }
            }

            IloLinearNumExpr nbAisles = cplex.linearNumExpr();
            for (int a = 0; a < aisles.size(); a++) {
                nbAisles.addTerm(1, aisleActivation[a]);
            }

            while (maxAisles > 0 && getRemainingTime(stopWatch) > 0) {

                // ======================= FIRST STAGE: Maximize the sum of quantity variables =======================
                cplex.addMaximize(nbItemsCollected);
                cplex.addLe(nbAisles, maxAisles, "MaximumAisles");
                cplex.setParam(IloCplex.Param.TimeLimit, getRemainingTime(stopWatch));
                if (!cplex.solve()) {
                    //System.out.println("No feasible solution found in the first stage with " + maxAisles + " aisles.");
                    break;
                }
                maxQuantity = Math.round(cplex.getObjValue());
                //System.out.println("===============>> First stage solution with " + maxAisles + " aisles = " + maxQuantity + " items.");
                cplex.delete(cplex.getObjective());

                // ================= SECOND STAGE: Adjust the model for minimizing the number of active aisles ===============
                IloRange minQuantityConstraint = cplex.addGe(nbItemsCollected, maxQuantity, "MinimumQuantity");
                cplex.addMinimize(nbAisles);
                cplex.setParam(IloCplex.Param.TimeLimit, getRemainingTime(stopWatch));
                if (!cplex.solve()) {
                    //System.out.println("No feasible solution found in the second stage.");
                    break;
                }
                double activeAisles = Math.round(cplex.getObjValue());
                //System.out.println("===============>> Second stage solution: Active aisles = " + activeAisles);
                

                // ================= UPDATE BEST SOLUTION ===============
                // Calculate the ratio and update the best solution
                double ratio = maxQuantity / activeAisles;
                if (ratio > bestRatio) {
                    bestRatio = ratio;
                    bestSolution = extractSolution(cplex, orderActivation, aisleActivation);
                    //System.out.println("New best solution found with ratio = " + bestRatio);
                    
                }
                if (activeAisles == 1) {
                    //System.out.println("Only one aisle was used.");
                    cplex.end();
                    return bestSolution;
                }
                // Update maxAisles for the next iteration
                maxAisles = (int) activeAisles - 1;
                cplex.delete(cplex.getObjective());
                cplex.delete(minQuantityConstraint);
            }

            // Clean up
            cplex.end();

            return bestSolution;

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private Map<String, IloNumVar> createQuantityVariables(IloCplex cplex) throws IloException {
        Map<String, IloNumVar> quantity = new HashMap<>();
        for (int o = 0; o < orders.size(); o++) {
            for (int i = 0; i < nItems; i++) {
                for (int a = 0; a < aisles.size(); a++) {
                    if (aisles.get(a).containsKey(i) && orders.get(o).containsKey(i)) {
                        String key = "quantity_" + i + "_" + o + "_" + a;
                        quantity.put(key, cplex.numVar(0, Double.MAX_VALUE, key));
                    }
                }
            }
        }
        return quantity;
    }

    private void addConstraints(IloCplex cplex, Map<String, IloNumVar> quantity, IloIntVar[] orderActivation, IloIntVar[] aisleActivation) throws IloException {
        
        // Constraints 1 and 2: Wave size bounds
        IloLinearNumExpr quantity_sum = cplex.linearNumExpr();
        for(int o = 0; o < orders.size(); o++){
            for (Map.Entry<Integer, Integer> entry : orders.get(o).entrySet()) {
                int i = entry.getKey();
                for(int a = 0; a < aisles.size(); a++){
                    String key = "quantity_" + i + "_" + o + "_" + a;
                    if (quantity.containsKey(key)) {
                        quantity_sum.addTerm(1., quantity.get(key));
                    }
                }
            }
        }
        cplex.addGe(quantity_sum, waveSizeLB, "LowerBoundQuantity");
        cplex.addLe(quantity_sum, waveSizeUB, "UpperBoundQuantity");

        // Constraint 3: Aisle item availability
        // items from all orders collected on an aisle must not be greater the item availability in the aisle 
        for(int a = 0; a < aisles.size(); a++){
            for (Map.Entry<Integer, Integer> entry : aisles.get(a).entrySet()) { // For each product on the aisle
                IloLinearNumExpr aisle_sum = cplex.linearNumExpr();
                int i = entry.getKey();
                int itemQuantity = entry.getValue();
                for(int o = 0; o < orders.size(); o++){ // For each order
                    if (orders.get(o).containsKey(i)){
                        String key = "quantity_" + i + "_" + o + "_" + a;
                        if (quantity.containsKey(key)) {
                            aisle_sum.addTerm(1., quantity.get(key));
                        }
                    }
                }
                cplex.addLe(aisle_sum, itemQuantity, "AvailabilityAisle_" + a + "_Item_" + i);
            }
        }
        
        // Constraint 4: Order activation
        for(int o = 0; o < orders.size(); o++){
            IloLinearNumExpr order_sum = cplex.linearNumExpr();
            for (Map.Entry<Integer, Integer> entry : orders.get(o).entrySet()) {
                int i = entry.getKey();
                for(int a = 0; a < aisles.size(); a++){
                    String key = "quantity_" + i + "_" + o + "_" + a;
                    if (quantity.containsKey(key)) {
                        order_sum.addTerm(1., quantity.get(key));
                    }
                }
                IloLinearNumExpr lhs = cplex.linearNumExpr();
                lhs.addTerm(waveSizeUB, orderActivation[o]);
                cplex.addGe(lhs, order_sum, "ActivationOrder_" + o);
            }
        }

        // Constraint 5: Aisle activation
        for(int a = 0; a < aisles.size(); a++){
            IloLinearNumExpr aisle_sum = cplex.linearNumExpr();
            for (Map.Entry<Integer, Integer> entry : aisles.get(a).entrySet()) {
                int i = entry.getKey();
                for(int o = 0; o < orders.size(); o++){
                    if (orders.get(o).containsKey(i)){
                        String key = "quantity_" + i + "_" + o + "_" + a;
                        if (quantity.containsKey(key)) {
                            aisle_sum.addTerm(1., quantity.get(key));
                        }
                    }
                }
            }
            IloLinearNumExpr lhs = cplex.linearNumExpr();
            lhs.addTerm(waveSizeUB, aisleActivation[a]);
            cplex.addGe(lhs, aisle_sum, "ActivationAisle_" + a);
        }

        // Constraint 6: Force to get all items from a selected order
        // For each item in each order, the sum in all aisles is equal the requested
        for(int o = 0; o < orders.size(); o++){
            for (Map.Entry<Integer, Integer> entry : orders.get(o).entrySet()) {
                int i = entry.getKey();
                int itemQuantity = entry.getValue();
                IloLinearNumExpr itemOrder_sum = cplex.linearNumExpr();
                IloLinearNumExpr itemOrder_min = cplex.linearNumExpr();
                itemOrder_min.addTerm(itemQuantity, orderActivation[o]);
                for(int a = 0; a < aisles.size(); a++){
                    String key = "quantity_" + i + "_" + o + "_" + a;
                    if (quantity.containsKey(key)) {
                        itemOrder_sum.addTerm(1., quantity.get(key));
                    }
                }
                cplex.addEq(itemOrder_sum, itemOrder_min, "ForceItemsOrder_" + o);
            }
        }

    }

    private ChallengeSolution extractSolution(IloCplex cplex, IloIntVar[] orderActivation, IloIntVar[] aisleActivation) throws IloException {
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> visitedAisles = new HashSet<>();
        for (int o = 0; o < orders.size(); o++) {
            if (cplex.getValue(orderActivation[o]) > 0.5) {
                selectedOrders.add(o);
            }
        }
        for (int a = 0; a < aisles.size(); a++) {
            if (cplex.getValue(aisleActivation[a]) > 0.5) {
                visitedAisles.add(a);
            }
        }
        return new ChallengeSolution(selectedOrders, visitedAisles);
    }
}
