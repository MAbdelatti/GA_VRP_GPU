import math
from numba import vectorize, guvectorize, jit, float32
import numpy as np

#@jit
#@guvectorize([(float32[:],float32[:,:], float32, float32[:,:])], '(m),(n,p),()->(n,n)')
#def adjust(individual, vrp_data, vrp_capacity, individual_mod):
def adjust(individual, vrp_data, vrp_capacity):
    # Create TEMP list to handle insert and remove of items (not supported for arrayes in GPU!!)
    # Adjust repeated
    repeated = True
    while repeated:
        repeated = False
        for i1 in range(len(individual)):
            # print('\n main',individual[i1],':',end=' ')
            for i2 in range(i1):
                # print(individual[i2], end=' ')
                if individual[i1] == individual[i2]:
                    haveAll = True
                    for i3 in range(len(vrp_data)):
                        nodeId = vrp_data[i3][0]
                        if nodeId not in individual: # ensure that All nodes (with demand > 0) are covered in each sigle solution
                            individual[i1] = nodeId
                            haveAll = False
                            break
                    if haveAll:
                        mask = np.ones(len(individual), dtype=bool)
                        mask[i1] = False
                        print(mask)
                        individual = individual[mask]
                    repeated = True
                if repeated: break
            if repeated: break
    # Adjust capacity exceed
    i = 0               # index
    reqcap = 0.0        # required capacity
    # print('\n ##############')
    while i < len(individual)-1: 
        reqcap += vrp_data[vrp_data[:,0] == individual[i]][0,1] if individual[i] !=0 else 0.0
        if reqcap > vrp_capacity: 
            individual = np.insert(individual, i, np.float32(0))
            reqcap = 0.0
        i += 1
    i = len(individual) - 2

    # Adjust two consective depots
    while i >= 0:
        if individual[i] == 0 and individual[i + 1] == 0:
            mask = np.ones(len(individual), dtype=bool)
            mask[i] = False
            individual = individual[mask]
        i -= 1
#    individual_mod[0] = individual[:4]
    # print('individual in adjust func: ',individual)
    # print('##############')
    return(individual)

vrp_capacity = 30 # Temporarily!!
#popsize = 5  # Temporarily!!
#iterations = 5  # Temporarily!!
vrp_data = np.array([[ 1.0,  7.0,  37.0,  52.0], [ 2.0,  30.0,  49.0,  49.0], [ 3.0, 16.0, 52.0, 64.0]], dtype=np.float32)  # Temporarily!!
individual = np.array(([3, 2, 1, 110.3]), dtype=np.float32)

x = adjust(individual, vrp_data, vrp_capacity)
print(x)