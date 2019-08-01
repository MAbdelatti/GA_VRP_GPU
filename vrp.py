import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import cProfile
from tqdm import tqdm

# from numba import float32, int64
# from numba import vectorize, guvectorize, jit, cuda

from timeit import default_timer as timer
import two_opt

pr = cProfile.Profile()
pr.enable

class node():
    def __init__(self, label, demand, posX, posY):
        self.label = label
        self.demand = demand
        self.X = posX
        self.Y = posY

class vrp():
    def __init__(self, capacity=None):
        self.capacity = capacity
        self.nodes = np.zeros((1,4), dtype=np.float32)

    def addNode(self, label, demand, posX, posY):
        newrow = np.array([label, demand, posX, posY], dtype=np.float32)
        self.nodes = np.vstack((self.nodes, newrow))

pop = []

def readInput():
	# Create VRP object:
    vrpManager = vrp()
	## First reading the VRP from the input ##
    print('Reading data file...', end=' ')
    fo = open(sys.argv[3],"r")
    lines = fo.readlines()
    for i, line in enumerate(lines):
        while line.upper().startswith('CAPACITY'):
            inputs = line.split()
            vrpManager.capacity = np.float32(inputs[2])
			# Validating positive non-zero capacity
            if vrpManager.capacity <= 0:
                print(sys.stderr, 'Invalid input: capacity must be neither negative nor zero!')
                exit(1)
            break       
        while line.upper().startswith('NODE_COORD_SECTION'):
            i += 1
            line = lines[i]
            while not (line.upper().startswith('DEMAND_SECTION') or line=='\n'):
                inputs = line.split()
                vrpManager.addNode(np.int16(inputs[0]), 0.0, np.float32(inputs[1]), np.float32((inputs[2])))
                # print(vrpManager.nodes)
                i += 1
                line = lines[i]
                while (line=='\n'):
                    i += 1
                    line = lines[i]
                    if line.upper().startswith('DEMAND_SECTION'): break 
                if line.upper().startswith('DEMAND_SECTION'):
                    i += 1
                    line = lines[i] 
                    while not (line.upper().startswith('DEPOT_SECTION')):                  
                        inputs = line.split()
						# Validating demand not greater than capacity
                        if float(inputs[1]) > vrpManager.capacity:
                            print(sys.stderr,
							'Invalid input: the demand of the node %s is greater than the vehicle capacity!' % vrpManager.nodes[0])
                            exit(1)
                        if float(inputs[1]) < 0:
                            print(sys.stderr,
                            'Invalid input: the demand of the node %s cannot be negative!' % vrpManager.nodes[0])
                            exit(1)                            
                        vrpManager.nodes[int(inputs[0])][1] =  float(inputs[1])
                        i += 1
                        line = lines[i]
                        while (line=='\n'):
                            i += 1
                            line = lines[i]
                            if line.upper().startswith('DEPOT_SECTION'): break
                        if line.upper().startswith('DEPOT_SECTION'):
                            vrpManager.nodes = np.delete(vrpManager.nodes, 0, 0)                          
                            print('Done.')
                            return(vrpManager.capacity, vrpManager.nodes)

def filter_out(vrp_capacity, vrp_data):
    # Temporarily, drop the nodes with (demands + any other demand) > capcity
    # Will be added after the GA finshes at the end
    dropped_nodes = [vrp_data[0].tolist()]
    dropped_routes = []
    for node in vrp_data[1:]:
        node_capacity = np.add([[0, node[1], 0, 0]]*(vrp_data.shape[0]-1), vrp_data[1:])
        if node_capacity[:,1].min() > vrp_capacity:
            # Aha! a node capacity + another capacity > vrp_capacity, drop it temporariy
            vrp_data = np.delete(vrp_data, np.where(vrp_data[:,0]==node[0]),0)
            dropped_nodes.append(node.tolist())
            dropped_routes.extend([1, node[0]])
    dropped_routes.append(1.0)
    return(vrp_data, dropped_nodes, dropped_routes)

def distance(first_node, prev, next_node, last_node, individual, vrp_data):
    total_dist = 0
	# The first distance is from depot to the first node of the first route
    if individual[0] != 1:
        for k in range(len(vrp_data)):
            if vrp_data[k][0] == individual[0]:
                first_node = vrp_data[k]
                break
    else:
        first_node = vrp_data[0]

    x1 = vrp_data[0][2]
    x2 = first_node[2]
    y1 = vrp_data[0][3]
    y2 = first_node[3]

    dx = x1 - x2
    dy = y1 - y2
    total_dist = (round(math.sqrt(dx * dx + dy * dy)))
		
	# Then calculating the distances between the nodes
    for i in range(len(individual) - 2):
        if individual[i] != 1:
            for k in range(len(vrp_data)):
                if vrp_data[k][0] == individual[i]:
                    prev = vrp_data[k]
                    break
        else:
            prev = vrp_data[0]

        if individual[i+1] != 1:
            for k in range(len(vrp_data)):
                if vrp_data[k][0] == individual[i+1]:
                    next_node = vrp_data[k]
                    break
        else:
            next_node = vrp_data[0]

        x1 = prev[2]
        x2 = next_node[2]
        y1 = prev[3]
        y2 = next_node[3]

        dx = x1 - x2
        dy = y1 - y2
        total_dist += (round(math.sqrt(dx * dx + dy * dy)))

	# The last distance is from the last node of the last route to the depot

    last_node = next_node

    x1 = last_node[2]
    x2 = vrp_data[0][2]
    y1 = last_node[3]
    y2 = vrp_data[0][3]
    dx = x1 - x2
    dy = y1 - y2
    total_dist += (round(math.sqrt(dx * dx + dy * dy)))
    return(total_dist)

def fitness(cost_table, individual):
    # nodes represent the row/column index in the cost table
    zeroed_indiv = np.subtract(individual, [1]*len(individual))
    if zeroed_indiv[0] != 0:
        zeroed_indiv = np.insert(zeroed_indiv,0,0)
    if individual[-1] != 1:
        zeroed_indiv = np.hstack((zeroed_indiv, 0))

    fitness_val = 0
    for i in range(len(zeroed_indiv)-1):
        fitness_val += cost_table[int(zeroed_indiv[i]), int(zeroed_indiv[i+1])]
    return fitness_val

def fitness_old(vrp_data, individual):
    first_node = np.zeros(4, dtype=np.float32)
    prev = np.zeros(4, dtype=np.float32)
    next_node = np.zeros(4, dtype=np.float32)
    last_node = np.zeros(4, dtype=np.float32)

    totaldist = distance(first_node, prev, next_node, last_node, individual, vrp_data)
    # no_of_vehicles = list(individual).count(1)

    return(totaldist)

def adjust(individual, vrp_data, vrp_capacity):
    # Delete duplicate nodes
    individual = individual.tolist()
    individual = sorted(set(individual), key=individual.index)

    # Check the missing nodes and insert them randomly
    missing_nodes = set(vrp_data[:,0]) - set(individual)

    for node in missing_nodes:
        individual.insert(random.randint(0, len(individual)-2), node)
    # Delete ones
    individual.remove(1)

    i = 0               # index
    reqcap = 0.0        # required capacity

    while i < len(individual)-1:
        reqcap += vrp_data[vrp_data[:,0] == individual[i]][0,1] if individual[i] != 1 else 0.0
        if reqcap > vrp_capacity: 
            individual = np.insert(individual, i, np.float32(1))
            reqcap = 0.0
        i += 1

    return individual
    
# Generating random initial population
def initializePop(vrp_data, cost_table, popsize, vrp_capacity):
    print('GA evolving, please wait until finished...')
    popArr = []
    nodes = []
    nodes += [float(node[0]) for node in vrp_data]
    for i in range(0, popsize):
        individual = nodes.copy()
        random.shuffle(individual)
        individual.append(9999.0) # Any number != 1
        individual = adjust(np.asarray(individual, dtype=np.float32), np.asarray(vrp_data, dtype=np.float32), vrp_capacity)
        fitness_val = fitness(cost_table, individual[:-1])
        individual[-1] = fitness_val
        individual = list(individual)
        individual.insert(0, 0)
        popArr += [individual]
    print('Initial population:\n', popArr)
    return(popArr)

def evolvePop(pop, vrp_data, iterations, popsize, vrp_capacity, extended_cost, opt, cost_table=0):
    # Running the genetic algorithm
    run_time = timer()
    stucking_indicator = 0
    for i in tqdm(range(iterations)):
        old_best = pop[0][-1]
        nextPop = []
        nextPop_set = set()

        elite_count = len(pop)//20      
        sorted_pop = pop.copy()

        # Apply two-opt for the new top 5% individuals:
        for idx, individual in enumerate(sorted_pop[:elite_count]):
            if individual[0] >= i:
                sorted_pop[idx], cost = two_opt.two_opt(individual[1:-1], cost_table)
                sorted_pop[idx].append(9999)
                fitness_value = fitness(cost_table, sorted_pop[idx][:-1])
                sorted_pop[idx][-1] = (fitness_value)
                sorted_pop[idx].insert(0,individual[0])
        
        sorted_pop.sort(key= lambda elem: elem[-1])
        pop = sorted_pop.copy()
        
        start_evolution_timer = timer()
        # terminate if optimal is reached or runtime exceeds 1h
        if ((sorted_pop[0][-1] + extended_cost) > opt) and (timer() - run_time <= 60):
            nextPop = sorted_pop[:elite_count] # top 5% of the parents will remain in the new generation         

            # for j in range(round(((len(pop))-elite_count) / 2)):
            while len(nextPop_set) < popsize:
                # Selecting randomly 4 individuals to select 2 parents by a binary tournament
                parentIds = set()
                while len(parentIds) < 4:
                    parentIds.add(random.randint(0, len(pop) - 1))

                # Avoid stucking to a local minimum swap after 25 generations of identical fitness
                #if stucking_indicator >= 25:
                    #print('\nstucking is spotted', pop[1])
                    #for idx, swapped_indiv in enumerate(pop[1:elite_count]):
                        #i1 = swapped_indiv[1:round(len(swapped_indiv)/2)]
                        #i2 = swapped_indiv[round(len(swapped_indiv)/2): -1]
                        ## i1 = random.randint(1, len(swapped_indiv) - 2)
                        ## i2 = random.randint(1, len(swapped_indiv) - 2)
                        #swapped_indiv = i2
                        #swapped_indiv = np.append(swapped_indiv, i1)
                        ## swapped_indiv[i1], swapped_indiv[i2] = swapped_indiv[i2], swapped_indiv[i1]
                        #swapped_indiv = adjust(np.asarray(swapped_indiv[1:], dtype=np.float32), np.asarray(vrp_data, dtype=np.float32), vrp_capacity)
                        #fitness_val = fitness(np.asarray(vrp_data, np.float32), np.asarray(swapped_indiv[1:], np.float32))
                        ## swapped_indiv[-1] = fitness_val
                        #swapped_indiv = np.append(swapped_indiv, fitness_val)
                        #pop[idx] = swapped_indiv
                    #stucking_indicator = 0
               
                parentIds = list(parentIds)
                # Selecting 2 parents with the binary tournament
                parent1 = list(pop[parentIds[0]] if pop[parentIds[0]][len(pop[parentIds[0]])-1] < pop[parentIds[1]][len(pop[parentIds[1]])-1] else pop[parentIds[1]])
                parent2 = list(pop[parentIds[2]] if pop[parentIds[2]][len(pop[parentIds[2]])-1] < pop[parentIds[3]][len(pop[parentIds[3]])-1] else pop[parentIds[3]])

                child1 = parent1[1:].copy()
                child2 = parent2[1:].copy()

                # Performing Two-Point crossover and generating two children
                # Selecting (n/5 - 1) random cutting points for crossover, with the same points (indexes) for both parents, based on the shortest parent

                cutIdx = [0] * ((min(len(parent1) - 2, len(parent2) - 2))//5 - 1)
                for k in range(0, len(cutIdx)):
                    cutIdx[k] = random.randint(1, min(len(parent1) - 2, len(parent2) - 2))
                    while cutIdx[k] in cutIdx[:k]:
                        cutIdx[k] = random.randint(1, min(len(parent1) - 2, len(parent2) - 2))
                cutIdx.sort()
                for k in range(0, len(cutIdx), 2):
                    if len(cutIdx) %2 == 1 and k == len(cutIdx) - 1: # Odd number
                        child1[cutIdx[k]:] = child2[cutIdx[k]:]
                        child2[cutIdx[k]:] = child1[cutIdx[k]:]
                    else:                       
                        child1[cutIdx[k]:cutIdx[k + 1]] = child2[cutIdx[k]:cutIdx[k + 1]]
                        child2[cutIdx[k]:cutIdx[k + 1]] = child1[cutIdx[k]:cutIdx[k + 1]]        

                # Doing mutation: swapping two positions in one of the individuals, with 1:15 probability
                mutation_prob = 40
                if random.randint(1, mutation_prob) == 1:
                    # Random swap mutation
                    ptomutate = child1
                    i1 = random.randint(0, len(ptomutate) - 2)
                    i2 = random.randint(0, len(ptomutate) - 2)
                    # Repeat random selection if depot was selected
                    while ptomutate[i1] == 1:
                        i1 = random.randint(0, len(ptomutate) - 2)
                    while ptomutate[i2] == 1:
                        i2 = random.randint(0, len(ptomutate) - 2)
                    ptomutate[i1], ptomutate[i2] = ptomutate[i2], ptomutate[i1]

                if random.randint(1, mutation_prob) == 1:
                    ptomutate = child2
                    i1 = random.randint(0, len(ptomutate) - 2)
                    i2 = random.randint(0, len(ptomutate) - 2)
                    # Repeat random selection if depot was selected
                    while ptomutate[i1] == 1:
                        i1 = random.randint(0, len(ptomutate) - 2)
                    while ptomutate[i2] == 1:
                        i2 = random.randint(0, len(ptomutate) - 2)
                    ptomutate[i1], ptomutate[i2] = ptomutate[i2], ptomutate[i1]

                # Adjusting individuals               
                child1 = adjust(np.asarray(child1, dtype=np.float32), np.asarray(vrp_data, dtype=np.float32), vrp_capacity)
                child2 = adjust(np.asarray(child2, dtype=np.float32), np.asarray(vrp_data, dtype=np.float32), vrp_capacity)

                # # Apply 2-opt:
                # child1, fitness_val = two_opt.two_opt(child1[:-1], cost_table)
                # child2, fitness_val = two_opt.two_opt(child2[:-1], cost_table)

                fitness_val = fitness(cost_table, child1[:-1])
                child1[-1] = fitness_val
                
                fitness_val = fitness(cost_table, child2[:-1])
                child2[-1] = fitness_val

                child1 = list(child1)
                child2 = list(child2)

                child1.insert(0, i + 1)
                child2.insert(0, i + 1)

                # Add children to population iff they are better than parents
                if (child1[-1] < parent1[-1]) | (child1[-1] < parent2[-1]) | ((timer() - start_evolution_timer) > 30):
                    nextPop_set.add(tuple(child1))
                    # start_evolution_timer = timer()
                    # nextPop_set.add(tuple(parent1))
                
                if (child2[-1] < parent1[-1]) | (child2[-1] < parent2[-1]) | ((timer() - start_evolution_timer) > 30):
                    nextPop_set.add(tuple(child2))
                    # start_evolution_timer = timer()
                    # nextPop_set.add(tuple(parent2))   
                               
            nextPop = list(nextPop_set)

            # Updating population generation

            # random.shuffle(nextPop)
            nextPop = sorted(nextPop, key= lambda elem: elem[-1])

            if nextPop[0][-1] == old_best:
                stucking_indicator += 1
            else:
                stucking_indicator = 0

            pop = nextPop
            if not (i+1) % 5: # print population every 300 generations
                print(f'Population at generation {i+1}:{pop}\nBest: {pop[0][-1]}')
        elif (timer() - run_time >= 60):
            print('Time criteria is met')
            break
        elif (((sorted_pop[0][-1] + extended_cost) <= opt)):
            print('Cost criteria is met')
            break
    return (pop)

vrp_capacity, data = readInput()
vrp_data, dropped_nodes, dropped_routes = filter_out(vrp_capacity, data)

## Calculate cost table:
cost_table = np.zeros((data.shape[0],data.shape[0]), dtype=np.float32)
vrp_data_for_cost = data.copy()
vrp_data_for_cost[:,0] = np.subtract(data[:,0], [1]*len(data[:,0]))

for index, node in enumerate(vrp_data_for_cost[:,0]):
    cost_table[index, index+1:] = np.round(np.hypot(np.subtract([vrp_data_for_cost[index,2]]*len(vrp_data_for_cost[index+1:, 2]), vrp_data_for_cost[index+1:, 2]),\
         np.subtract([vrp_data_for_cost[index,3]]*len(vrp_data_for_cost[index+1:, 3]),vrp_data_for_cost[index+1:, 3])))

cost_table =  np.add(cost_table, np.transpose(cost_table))
# ------

if len(dropped_nodes) > 1:
    extended_cost = fitness(cost_table, dropped_routes)
else:
    extended_cost = 0

popsize = int(sys.argv[1])
iterations = int(sys.argv[2])
opt = 0.0 if len(sys.argv) == 4 else int(sys.argv[4])

import multiprocessing as MLP
from concurrent.futures import ThreadPoolExecutor

cpu_no = MLP.cpu_count()
pool = ThreadPoolExecutor(max_workers=cpu_no)

start = timer()
# pop = initializePop(vrp_data, cost_table, popsize, vrp_capacity)
future_1 = pool.submit(initializePop, vrp_data, cost_table, popsize, vrp_capacity)
pop = future_1.result()

# pop = evolvePop(pop, vrp_data, iterations, popsize, vrp_capacity, extended_cost, opt, cost_table)
future_2 = pool.submit(evolvePop, pop, vrp_data, iterations, popsize, vrp_capacity, extended_cost, opt, cost_table)
pop = future_2.result()

# Selecting the best individual, which is the final solution
better = []
individual = min(pop, key= lambda idx: idx[len(idx) - 1])
individual = list(individual)

# Add the dropped routes & cost to the end of the best solution
individual[-1:-1] = dropped_routes[:-1]
if len(dropped_nodes) > 1:
    individual[-1] += extended_cost
better = [1] + list(individual[1:-1]) if individual[1] != 1 else list(individual[1:-1])

data = np.subtract(data, [[1, 0, 0, 0]]*data.shape[0])
better = list(np.subtract(better, [1]*len(better)))

t = int(timer()-start)

# Printing & plotting solution
print ('Solution by GA:\n',  better)
print ('Cost:', individual[-1])
print('Solved at Generation:', individual[0])

final_pop = pop
for i in range(len(final_pop)):
    final_indiv = list(final_pop[i])
    final_indiv[-1] += extended_cost
    final_pop[i] = list(np.subtract(final_indiv[1:-1], [1]*len(final_indiv[1:-1])))
    final_pop[i].insert(0,final_indiv[0])
    final_pop[i].append(final_indiv[-1])

print('Final population:\n', final_pop)

# Plot solution:
plt.scatter(data[1:][:,2], data[1:][:,3], c='b')
plt.plot(data[0][2], vrp_data[0][3], c='r', marker='s')

line_1 = None
# for loc, i in enumerate(better):
#     if i != 1:
#         # Text annotations for data points:
#         plt.annotate(('%d\n"%d"'%(i, data[data[:,0]==i][0][1])), (data[data[:,0]==i][0][2]+1,data[data[:,0]==i][0][3]))
#     if loc != len(better)-1:
#         # Plot routes

#         plt.plot([data[data[:,0]==i][0][2], data[data[:,0]==better[loc+1]][0][2]],\
#          [data[data[:,0]==i][0][3], data[data[:,0]==better[loc+1]][0][3]]\
#              , c='k', linestyle='--', alpha=0.3)
#     else:
#         line_1, = plt.plot([data[data[:,0]==i][0][2], data[0][2]],\
#          [data[data[:,0]==i][0][3], data[0][3]], label='GA only: %d'%individual[-1]\
#              , c='k', linestyle='--', alpha=0.3)

# plt.axis('equal')

# Solve routes as TSP:
# import two_opt
# route = [0,	38,	9,	29,	21,	34,	30,	10,	39,	33,	15,	0	,12,	5,	37,	17,	19,	13,	4	,0,	18	,25	,14	,24,	23,	6,	0,	11,	16,	2,	20,	35,	36,	3,	1,	0,	27,	32,	22,	28,	31,	8,	26,	7]

# sequence, cost = two_opt.two_opt(route, cost_table)

# # print('After 2-opt', sequence, cost)

# import tsp_cplex as tsp
# tsp.solve(better, data, line_1)