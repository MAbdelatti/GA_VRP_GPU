import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt

import sys
import os

def solve(better, vrp_data):
    # better = [0, 3.0, 9.0, 10.0, 0.0, 6.0, 0.0, 13.0, 7.0, 5.0, 0.0, 4.0, 0.0, 2.0, 0.0, 12.0, 11.0, 15.0, 0.0, 14.0, 0.0, 1.0, 0.0, 8.0]
    print(better)
    print('Solving the routes as TSP...')   
    # Divide solution into routes:
    idx_lo= 0
    routes = []

    for i, val in enumerate(better):
        if (val == 0 and i!=0): 
            idx_hi = i 
            routes.append(better[idx_lo:idx_hi])
            idx_lo = idx_hi
    routes.append(better[idx_hi:])
    
    cost = 0
    sorted_best = []
    for route in routes:
        active_arcs = []
        # Problem definition
        n = len(route) # Total number of cities

        # route = range(0, n)
        # route = [0, 31, 1, 3, 51, 40, 10, 11, 4, 22, 5, 6, 9, 7, 15, 2]
        
        # Optimize routes that have more than 2 cities
        if len(route) <= 2:
            route.sort()
            sorted_best.append(route[0])
            sorted_best.append(route[1])
        else:
            loc_x, loc_y = [], []
            for i in route:
                loc_x.append(vrp_data[vrp_data[:,0]==i][0][2])
                loc_y.append(vrp_data[vrp_data[:,0]==i][0][3])

            X = set([(i, j) for i in route for j in route if i!=j]) # List of Arcs
            c = {(i,j): round(np.hypot(loc_x[route.index(i)]-loc_x[route.index(j)], loc_y[route.index(i)]-loc_y[route.index(j)])) for i, j in X} # Dictionary of distances/costs

            # Create a CPLEX model:
            mdl = Model('TSP')


            # Define arcs and capacities:
            x = mdl.binary_var_dict(X, name= 'x')
            # u = mdl.continuous_var_list(n, 0, float('inf'))
            u = mdl.continuous_var_dict(route, 0, float('inf'))

            # Define objective function:
            mdl.minimize(mdl.sum(c[i,j]*x[i,j] for i, j in X if i!=j))

            # Add constraints:
            mdl.add_constraints(mdl.sum(x[i,j] for i in route if i != j and (i != 0 or j!=0)) == 1 for j in route) # Each point must be visited
            mdl.add_constraints(mdl.sum(x[i,j] for j in route if j != i and (i != 0 or j!=0)) == 1 for i in route) # Each point must be left
            mdl.add_indicator_constraints(mdl.indicator_constraint(x[i,j], u[i]-u[j]+(n-1)*x[i,j] <= n-2) for i, j in X if i!=0 and j!=0 and i!=j)
            # mdl.parameters.timelimit = 15 # Add running time limit

            # Solving model:
            solution = mdl.solve(log_output=False)
            # cost += solution.objective_value

            # print(solution)
            # print(solution.solve_status) # Returns if the solution is Optimal or just Feasible
            print('route cost:', solution.objective_value)
            
            active_arcs.append([a for a in X if x[a].solution_value > 0.9]) 

            # Plot solution:
            # plt.scatter(loc_x[1:], loc_y[1:], c='b')
            # for i in route:
            #     plt.annotate((route.index(i)), (loc_x[route.index(i)]+2,loc_y[route.index(i)]))

            # for i, j in active_arcs:
            #     plt.plot([loc_x[route.index(i)], loc_x[route.index(j)]], [loc_y[route.index(i)], loc_y[route.index(j)]], c='g', alpha=0.3)
            # plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
            # plt.axis('equal')
            # plt.show()
        # else:
        #     active_arcs.append((route))
        #     cost += round(np.hypot(loc_x[0]-loc_x[1], loc_y[0]-loc_y[1]))

            key = 0
            to_break = False
            print('active_arcs:', active_arcs[0])
            for i in range(len(active_arcs[0])):
                if to_break: break
                for element in active_arcs[0]:
                    if element[0] == key:
                        pair = element
                        sorted_best.append(pair[0])
                        key = pair[1]
                        if key == 0: to_break = True; break

    sorted_best.append(0)
    print('\nSolution after TSP optimization:\n', sorted_best)
    print('Cost after TSP optimization:', cost)

float32 = np.float32
better = [0, 3.0, 9.0, 10.0, 0.0, 6.0, 0.0, 13.0, 7.0, 5.0, 0.0, 4.0, 0.0, 2.0, 0.0, 12.0, 11.0, 15.0, 0.0, 14.0, 0.0, 1.0, 0.0, 8.0, 422]
vrp_data = np.array([np.array([0., 0., 0., 0.], dtype=float32), np.array([ 1., 19., 37., 52.], dtype=float32), np.array([ 2., 30., 49., 49.], dtype=float32), np.array([ 3., 16., 52., 64.], dtype=float32), np.array([ 4., 23., 31., 62.], dtype=float32), np.array([ 5., 11., 52., 33.], dtype=float32), np.array([ 6., 31., 42., 41.], dtype=float32), np.array([ 7., 15., 52., 41.], dtype=float32), np.array([ 8., 28., 57., 58.], dtype=float32), np.array([ 9.,  8., 62., 42.], dtype=float32), np.array([10.,  8., 42., 57.], dtype=float32), np.array([11.,  7., 27., 68.], dtype=float32), np.array([12., 14., 43., 67.], dtype=float32), np.array([13.,  6., 58., 48.], dtype=float32), np.array([14., 19., 58., 27.], dtype=float32), np.array([15., 11., 37., 69.], dtype=float32)])
solve(better[:-1], vrp_data)