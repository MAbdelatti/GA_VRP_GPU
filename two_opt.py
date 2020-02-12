import numpy as np
import random



def two_opt(individual, cost_table, min_cost = 0):
    
    # Divide solution into routes:
    idx_lo= 0
    routes = []
    individual = np.subtract(individual, [1]*len(individual))

    for i, val in enumerate(individual):
        if (val == 0 and i!=0): 
            idx_hi = i 
            routes.append(individual[idx_lo:idx_hi])
            idx_lo = idx_hi
    routes.append(individual[idx_hi:])

    total_cost = 0
    reordered_indiv = []
    
    for sequence in routes:    
        improved = True

    ##   load data by coordinate table     
        TT = cost_table

        # sequence = import_data_from_csv(sequence).astype(int)
        L = len(TT) # number of rows
        min_cost =0
        len_sequence =len(sequence)
        sequence = np.asarray(sequence, dtype=int)

        for s in range(0,len_sequence-1):
            min_cost = np.sum([min_cost ,TT[sequence[s],sequence[s+1]]])
            
        min_cost = np.sum([min_cost ,TT[sequence[len_sequence-1],sequence[0]]])
    #-------------
        R = np.ceil((len_sequence-2)/L).astype(int)

        for i in range(R):
            improved = True
            if i == 0:
                Start = 0
                End = L
                while improved:
                    improved = False
                    for ii in range(Start+1, End-2):
                        for j in range(ii+1, End):
                            if j-ii == 1: continue # changes nothing, skip then
                            new_route = 1*sequence[:]
                            
                            new_route[ii:j] = sequence[j-1:ii-1:-1] # this is the 2woptSwap
                            
                            #cost function
                            new_cost =0
                            for s in range(0,len_sequence-1):
                                new_cost = np.sum([new_cost ,TT[new_route[s],new_route[s+1]]])
                            new_cost = np.sum([new_cost ,TT[new_route[len_sequence-1],new_route[0]]])
                            # --------
                            if new_cost < min_cost:
                                sequence = 1*new_route
                                improved = True
                                min_cost = 1*new_cost
            
            elif i == R-1:
                Start = len_sequence - L+1 
                End = len_sequence
                while improved:
                    improved = False
                    for ii in range(Start+1, End-2):
                        for j in range(ii+1, End):
                            if j-ii == 1: continue # changes nothing, skip then
                            new_route = 1*sequence[:]
                            new_route[ii:j] = sequence[j-1:ii-1:-1] # this is the 2woptSwap
                            
                            #cost function
                            new_cost =0
                            for s in range(0,len_sequence-1):
                                new_cost = np.sum([new_cost ,TT[new_route[s],new_route[s+1]]])
                            new_cost = np.sum([new_cost ,TT[new_route[len_sequence-1],new_route[0]]])
                            # --------
                            if new_cost < min_cost:
                                sequence = 1*new_route
                                improved = True
                                min_cost = 1*new_cost
                
                

            else:
                Start = i*(L-1)+2
                End = (i+1)*(L-1)+1
                while improved:
                    improved = False
                    for ii in range(Start+1, End-2):
                        for j in range(ii+1, End):
                            if j-ii == 1: continue # changes nothing, skip then
                            new_route = 1*sequence[:]
                            new_route[ii:j] = sequence[j-1:ii-1:-1] # this is the 2woptSwap
                            
                            #cost function
                            new_cost =0
                            for s in range(0,len_sequence-1):
                                new_cost = np.sum([new_cost ,TT[new_route[s],new_route[s+1]]])
                            new_cost = np.sum([new_cost ,TT[new_route[len_sequence-1],new_route[0]]])
                            # --------
                            if new_cost < min_cost:
                                sequence = 1*new_route
                                improved = True
                                min_cost = 1*new_cost
        total_cost += min_cost
        reordered_indiv = np.append(reordered_indiv,sequence)
    reordered_indiv = np.add(reordered_indiv, [1]*len(reordered_indiv))
    reordered_indiv = list(reordered_indiv)
    return (reordered_indiv, total_cost)