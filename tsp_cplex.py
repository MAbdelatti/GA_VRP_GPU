import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt

import sys
import os

# Define a problem:
rnd = np.random
rnd.seed(0)

n = 16 # Total number of cities
# Q = 15
A = range(0, n)

# V = [0] + N
# q = {i:rnd.randint(1,10) for i in range(15)}

loc_x = rnd.rand(len(A))*200
loc_y = rnd.rand(len(A))*100

X = [(i, j) for i in A for j in A if i!=j] # List of Arcs
c = {(i,j): round(np.hypot(loc_x[i]-loc_x[j], loc_y[i]-loc_y[j])) for i, j in X} # Dictionary of distances/costs
# Create a CPLEX model:
mdl = Model('TSP')

# Define arcs and capacities:
x = mdl.binary_var_dict(X, name= 'x')
u = mdl.continuous_var_list(n, 0, float('inf'))

# Define objective function:
mdl.minimize(mdl.sum(c[i,j]*x[i,j] for i, j in X if i!=j))

# Add constraints:
mdl.add_constraints(mdl.sum(x[i,j] for i in A if i != j) == 1 for j in A) # Each point must be visited
mdl.add_constraints(mdl.sum(x[i,j] for j in A if j != i) == 1 for i in A) # Each point must be left
mdl.add_indicator_constraints(mdl.indicator_constraint(x[i,j], u[i]-u[j]+(n-1)*x[i,j] <= n-2) for i, j in X if i!=0 and j!=0 and i!=j)
# mdl.add_constraints(u[i] >= q[i] for i in N)
# mdl.parameters.timelimit = 15 # Add running time limit

# Solving model:
solution = mdl.solve(log_output=True)

# print(solution)
print(solution.solve_status) # Returns if the solution is Optimal or just Feasible

active_arcs = [a for a in X if x[a].solution_value > 0.9]
print(active_arcs)
