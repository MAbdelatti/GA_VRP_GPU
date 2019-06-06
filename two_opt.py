import numpy as np

def two_opt(sequence = 'better',  cost_table ='cost_table_by_coordinates.csv', vrp_data = 'vrp_data', line_1 = 'line_1', min_cost = 0):
     #route = sequence[:nodes]
    
    improved = True

##   load data by coordinate table     
    TT = import_data_from_csv(cost_table)

    sequence = import_data_from_csv(sequence).astype(int)
    L = len(TT)
    min_cost =0
    len_sequence =len(sequence)

    for s in range(0,len_sequence-1):
        min_cost = np.nansum([min_cost ,TT[sequence[s],sequence[s+1]]])
        
    min_cost = np.nansum([min_cost ,TT[sequence[len_sequence-1],sequence[0]]])
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
                            new_cost = np.nansum([new_cost ,TT[new_route[s],new_route[s+1]]])
                        new_cost = np.nansum([new_cost ,TT[new_route[len_sequence-1],new_route[0]]])
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
                            new_cost = np.nansum([new_cost ,TT[new_route[s],new_route[s+1]]])
                        new_cost = np.nansum([new_cost ,TT[new_route[len_sequence-1],new_route[0]]])
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
                            new_cost = np.nansum([new_cost ,TT[new_route[s],new_route[s+1]]])
                        new_cost = np.nansum([new_cost ,TT[new_route[len_sequence-1],new_route[0]]])
                        # --------
                        if new_cost < min_cost:
                            sequence = 1*new_route
                            improved = True
                            min_cost = 1*new_cost
            

    export_data_to_csv(sequence , 'NN_best_route_2opt.csv')
    
    return (sequence, min_cost)

print(two_opt(sequence = 'NN_best_route.csv'))