import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt

import sys
import os

def solve(better, vrp_data, line_1):
    line_2 = None
    print('\nSolving the routes as TSP...')
    vrp_data[0,0] = 0
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
        route = list(route)
        active_arcs = []
        # Problem definition
        n = len(route) # Total number of cities
      
        loc_x, loc_y = [], []
        for i in route:
            loc_x.append(vrp_data[vrp_data[:,0]==i][0][2])
            loc_y.append(vrp_data[vrp_data[:,0]==i][0][3])

        # List of Arcs
        X = set([(i, j) for i in route for j in route if i!=j])
        # Dictionary of distances/costs
        c = {(i,j): round(np.hypot(loc_x[route.index(i)]-loc_x[route.index(j)], loc_y[route.index(i)]-loc_y[route.index(j)])) for i, j in X} 

        # Optimize routes that have more than 2 cities only
        if len(route) == 2:
            route.sort()
            sorted_best.append(route[0])
            sorted_best.append(route[1])
            cost += c[route[0], route[1]] * 2
            plt.plot([vrp_data[vrp_data[:,0]==route[0]][0][2], vrp_data[vrp_data[:,0]==route[1]][0][2]],\
                     [vrp_data[vrp_data[:,0]==route[0]][0][3], vrp_data[vrp_data[:,0]==route[1]][0][3]], c='b', alpha=0.7)
        else:
            # Create a CPLEX model:
            mdl = Model('TSP')


            # Define arcs:
            x = mdl.binary_var_dict(X, name= 'x')
            u = mdl.continuous_var_dict(route, 0, float('inf'))

            # Define objective function:
            mdl.minimize(mdl.sum(c[i,j]*x[i,j] for i, j in X if i!=j))

            # Add constraints:
            mdl.add_constraints(mdl.sum(x[i,j] for i in route if i != j and (i != 0 or j!=0)) == 1 for j in route) # Each point must be visited
            mdl.add_constraints(mdl.sum(x[i,j] for j in route if j != i and (i != 0 or j!=0)) == 1 for i in route) # Each point must be left
            mdl.add_indicator_constraints(mdl.indicator_constraint(x[i,j], u[i]-u[j]+(n-1)*x[i,j] <= n-2) for i, j in X if i!=0 and j!=0 and i!=j)
            # mdl.parameters.timelimit = 15 # Add running time limit

            # Solving model
            solution = mdl.solve(log_output=False)
           
            try:
                cost += solution._objective
            except:
                print(solution)
                print('route:', route)
                print('cost:', cost)
            active_arcs.append([a for a in X if x[a].solution_value > 0.9]) 

            # Plot solution
            for i, j in active_arcs[0]:
                line_2, = plt.plot([loc_x[route.index(i)], loc_x[route.index(j)]], [loc_y[route.index(i)], loc_y[route.index(j)]],\
                    c='b', alpha=0.7)
            
            # Reshape solution (format only)
            key = 0
            to_break = False
            for i in range(len(active_arcs[0])):
                if to_break: break
                for element in active_arcs[0]:
                    if element[0] == key:
                        pair = element
                        sorted_best.append(pair[0])
                        key = pair[1]
                        if key == 0: to_break = True; break   

    sorted_best.append(0)
    print('Solution:\n', sorted_best)
    print('Cost:', cost)

    plt.legend(handles=[line_1, line_2],labels=[line_1.get_label(),'With TSP: %d'%cost])

    plt.show()

# float32 = np.float32
# better = [0, 3.0, 9.0, 10.0, 0.0, 6.0, 0.0, 13.0, 7.0, 5.0, 0.0, 4.0, 0.0, 2.0, 0.0, 12.0, 11.0, 15.0, 0.0, 14.0, 0.0, 1.0, 0.0, 8.0, 422]
# vrp_data = np.array([np.array([0., 0., 40., 40.], dtype=float32), np.array([ 1., 19., 37., 52.], dtype=float32), np.array([ 2., 30., 49., 49.], dtype=float32), np.array([ 3., 16., 52., 64.], dtype=float32), np.array([ 4., 23., 31., 62.], dtype=float32), np.array([ 5., 11., 52., 33.], dtype=float32), np.array([ 6., 31., 42., 41.], dtype=float32), np.array([ 7., 15., 52., 41.], dtype=float32), np.array([ 8., 28., 57., 58.], dtype=float32), np.array([ 9.,  8., 62., 42.], dtype=float32), np.array([10.,  8., 42., 57.], dtype=float32), np.array([11.,  7., 27., 68.], dtype=float32), np.array([12., 14., 43., 67.], dtype=float32), np.array([13.,  6., 58., 48.], dtype=float32), np.array([14., 19., 58., 27.], dtype=float32), np.array([15., 11., 37., 69.], dtype=float32)])
# solve(better[:-1], vrp_data)