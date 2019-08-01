import numpy as np
from numba import cuda, jit, int32, float32, int64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from math import pow, hypot, ceil
import random
from pdb import set_trace
########################################
class vrp():
    def __init__(self, capacity=None):
        self.capacity = capacity
        self.nodes = np.zeros((1,4), dtype=np.float32)
    def addNode(self, label, demand, posX, posY):
        newrow = np.array([label, demand, posX, posY], dtype=np.float32)
        self.nodes = np.vstack((self.nodes, newrow))

# Read the problem data file
def readInput():
	# Create VRP object:
    vrpManager = vrp()
	## First reading the VRP from the input ##
    print('Reading data file...', end=' ')
    fo = open('/home/conda_user/GA_VRP/test_set/P/P-n16-k8.vrp',"r")
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
#######################################################
## Calculate cost table:
@cuda.jit
def calc_cost_gpu(data_d, popsize, vrp_capacity, cost_table_d):

    threadId_row, threadId_col = cuda.grid(2)
    
#     data_d[threadId_row,0] = data_d[threadId_row,0] - 1
    
####ceil() is used instead of round() as the latter crashes the kernel.
####This causes +1 values in some cost distances

    if (threadId_row <= data_d.shape[0]-1) and (threadId_col <= data_d.shape[0]-1):
        cost_table_d[threadId_row, threadId_col] = ceil(hypot(data_d[threadId_row,2] - data_d[threadId_col,2],\
                                                              data_d[threadId_row,3] - data_d[threadId_col,3]))
#     popArr = initializePop(data, popsize, vrp_capacity, cost_table)
################################################################
# Generating random initial population
@cuda.jit
def initializePop_gpu(rng_states, data_d, pop_d):
    threadId_row, threadId_col = cuda.grid(2)
    
    # Generate the individuals from the nodes in data_d:
    if threadId_col <= pop_d.shape[1]-1:
        pop_d[threadId_row,threadId_col] = data_d[threadId_col, 0]
    
    index = threadId_row*(cuda.blockDim.x)+threadId_col       
    
    # Randonly shuffle each individual on a separate thread:   
    col = 0
    if threadId_row <= pop_d.shape[0]-1 and threadId_col <= data_d.shape[0]-1 and threadId_col != 0:
        while col == 0:
            col = int(xoroshiro128p_uniform_float32(rng_states, threadId_row*threadId_col)*(data_d.shape[0]-1)+1)

            pop_d[threadId_row, threadId_col], pop_d[threadId_row, col] =\
        pop_d[threadId_row, col], pop_d[threadId_row, threadId_col]
        
    # Adjust individuals using adjust_gpu function:
    # Calculate fitness of each individual using fitness_gpu function:
##################################################################################
@cuda.jit
def adjust_gpu(data_d, vrp_capacity, cost_table_d, missing_d, pop_d):
    
    # nodes represent the row/column index in the cost table
    threadId_row, threadId_col = cuda.grid(2)
    
    # Remove duplicated elements from every single individual/row in population array:
    r_flag = 9999 # A flag for removal/replacement
    
    if threadId_row <= pop_d.shape[0]-1 and threadId_col <= pop_d.shape[1]-1 and threadId_col != 0:
                    
        for i in range(threadId_col-1, -1, -1):
            if pop_d[threadId_row, threadId_col] == pop_d[threadId_row, i]\
            and pop_d[threadId_row, threadId_col] != 0:
                pop_d[threadId_row, threadId_col] = r_flag 
            
        for j in range(data_d.shape[0]):
            for i in range(threadId_col-1, -1, -1):
                if data_d[j,0] == pop_d[threadId_row, i]:
                    missing_d[threadId_row, j] = 0
                    break
                else:
                    missing_d[threadId_row, j] = data_d[j,0]
                     
    # Add missing nodes to every single individual:
            
    if threadId_col == pop_d.shape[1]-1:
        missing_elements = True
        for i in range(missing_d.shape[1]):
                if missing_d[threadId_row, i] != 0:
                    missing_elements = True
                    for j in range(pop_d.shape[1]):
                        if pop_d[threadId_row, j] == r_flag:
                            pop_d[threadId_row, j] = missing_d[threadId_row, i]
                            missing_d[threadId_row, i] = 0
                            break
                else:
                    missing_elements = False

        if not missing_elements:
        # shift individual's elements to the left for every inserted '1':
            for i in range(pop_d.shape[1], 0, -1):
                if pop_d[threadId_row, i] == r_flag:
                    for j in range(i, pop_d.shape[1]-1):
                        new_val = pop_d[threadId_row, j+1]
                        pop_d[threadId_row, j] = new_val

        reqcap = 0.0        # required capacity
        for i in range(pop_d.shape[1]-1):
            if pop_d[threadId_row, i] != 1 and pop_d[threadId_row, i] != 0:
                reqcap += data_d[pop_d[threadId_row, i]-1, 1]
                if reqcap > vrp_capacity:
#                     # here will be the insert '1' algorithm:
                    new_val = 1
                    rep_val = pop_d[threadId_row, i]
                    
#                     # shift individual's elements to the right for every inserted '1': 
                    for j in range(i, pop_d.shape[1]-1):
                        pop_d[threadId_row, j] = new_val
                        new_val = rep_val
                        rep_val = pop_d[threadId_row, j+1]
                    reqcap = 0.0                    
            else:
                reqcap = 0.0
                
            
            # The last part is to add the individual's fitness value at the very end of it.
#             pop_d[threadId_row, -1] = # individual's fitness value

#     while i < len(adjusted_indiv): 
#         if adjusted_indiv[i] != 1:
#             reqcap += data[data[:,0] == adjusted_indiv[i]][0,1]
#         else:
#             reqcap = 0
        
#         if reqcap > vrp_capacity: 
#             adjusted_indiv = np.hstack((adjusted_indiv[:i], np.array([1], dtype=np.int32), adjusted_indiv[i:]))
#             reqcap = 0.0
#         i += 1
        
#     if adjusted_indiv[0] != 1:
#         adjusted_indiv = np.hstack((np.array([1], dtype=np.int32), adjusted_indiv))
#     if adjusted_indiv[-1] != 1:
#         adjusted_indiv = np.hstack((adjusted_indiv, np.array([1], dtype=np.int32)))
    
# #     adjusted_indiv = np.hstack((adjusted_indiv, np.asarray([fitness(cost_table, adjusted_indiv)], dtype=np.int32)))
#     return adjusted_indiv
###########################################################################
vrp_capacity, data = readInput()
popsize = 100
generations = 7000

data_d = cuda.to_device(data)
cost_table_d = cuda.device_array(shape=(data.shape[0], data.shape[0]), dtype=np.int32)

pop = np.zeros((popsize, 2*data.shape[0]+2), dtype=np.int32)
pop_d = cuda.to_device(pop)

# GPU grid configurations:
threads_per_block = (10, 10)
blocks_no = (2*data.shape[0])*popsize/pow(threads_per_block[0],2)

blocks = (ceil(blocks_no), ceil(blocks_no))
rng_states = create_xoroshiro128p_states(threads_per_block[0]**2  * blocks[0]**2, seed=1)
calc_cost_gpu[blocks, threads_per_block](data_d, popsize, vrp_capacity, cost_table_d)

initializePop_gpu[blocks, threads_per_block](rng_states, data_d, pop_d)

print(pop_d.copy_to_host()[50:65,:])
# print(cost_table_d.copy_to_host())
###############################################################################################
# Speed test of CPU and GPU versions of the function:
# cost_table = np.zeros((data.shape[0],data.shape[0]), dtype=np.int32)
# print(calc_cost(data, popsize, vrp_capacity, cost_table).shape)
# print('CPU time:')
# %timeit calc_cost(data, popsize, vrp_capacity, cost_table)
# print('GPU time:')
# %timeit calc_cost_gpu[blocks, threads_per_block](data_d, popsize, vrp_capacity, cost_table_d)
################################################################################################
# zeros = np.zeros(individual_d.shape[0], dtype=np.int32)
# adjusted_indiv = cuda.to_device(zeros)
zeros = np.zeros(shape=(popsize, pop_d.shape[1]), dtype=np.int32)
missing_d = cuda.to_device(zeros)

print(pop_d.copy_to_host()[50:65,:], end='\n-----------------------\n')
#%timeit adjust_gpu[blocks,threads_per_block]\
 #(data_d, vrp_capacity, cost_table_d, missing_d, pop_d)
adjust_gpu[blocks,threads_per_block](data_d, vrp_capacity, cost_table_d, missing_d, pop_d)
print(missing_d.copy_to_host()[50:65,:])
print(pop_d.copy_to_host()[50:65,:])

# fitness_gpu[blocks,threads_per_block](cost_table_d, adjusted_indiv, zeroed_indiv_d, fitness_val_d)
# print(fitness_val_d.copy_to_host()[0])