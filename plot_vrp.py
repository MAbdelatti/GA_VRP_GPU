import matplotlib.pyplot as plt
import numpy as np
import sys

def read_data():
    nodes = np.zeros(4, dtype=float)
    # Read the data file:
    print('Reading data file...', end=' ')
    # problem_file = open("/home/marwan/research/GA_VRP/test_set/P/P-n16-k8.vrp","r")
    problem_file = open(sys.argv[1],"r")
    lines = problem_file.readlines()

    for i, line in enumerate(lines):
        while line.upper().startswith('CAPACITY'):
            inputs = line.split()
            vCapacity = float(inputs[2])
			# Validating positive non-zero capacity
            if vCapacity <= 0:
                print(sys.stderr, 'Invalid input: capacity must be neither negative nor zero!')
                exit(1)
            break       
        while line.upper().startswith('NODE_COORD_SECTION'):
            i += 1
            line = lines[i]
            while not (line.upper().startswith('DEMAND_SECTION') or line=='\n'):
                inputs = line.split()
                nodes = np.vstack((nodes, [int(inputs[0]), 0.0, float(inputs[1]), float((inputs[2]))]))
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
                        if float(inputs[1]) > vCapacity:
                            print(sys.stderr,
							'Invalid input: the demand of the node %s is greater than the vehicle capacity!' % nodes[0])
                            exit(1)
                        if float(inputs[1]) < 0:
                            print(sys.stderr,
                            'Invalid input: the demand of the node %s cannot be negative!' % nodes[0])
                            exit(1)                            
                        nodes[int(inputs[0])][1] =  float(inputs[1])
                        i += 1
                        line = lines[i]
                        while (line=='\n'):
                            i += 1
                            line = lines[i]
                            if line.upper().startswith('DEPOT_SECTION'): break
                        if line.upper().startswith('DEPOT_SECTION'):
                            nodes = np.delete(nodes,0,axis=0)
                            print('Done.')
                            return(nodes)

def read_sol_file():
    problem_file = open(sys.argv[2],"r")
    lines = problem_file.readlines()
    better = np.asarray(lines[0].split(','), dtype=np.float32)
    return(better)

def plot_sol():
    data = np.subtract(read_data(), np.asarray([1,0,0,0]))
    plt.scatter(data[1:][:,2], data[1:][:,3], c='b')            # plot the nodes
    plt.plot(data[0][2], data[0][3], c='r', marker='s')     # plot the depot

    better = read_sol_file()
    print(data[data[:,0]==better[4+1]])

    for loc, i in enumerate(better):
        if i != 0:
            # Text annotations for data points:
            # plt.annotate(('%d\n"%d"'%(i, data[data[:,0]==i][0][1])), (data[data[:,0]==i][0][2]+1,data[data[:,0]==i][0][3]))
            plt.annotate(int(i), (data[data[:,0]==i][0][2]+1,data[data[:,0]==i][0][3]))
        if loc != len(better)-1:
            # Plot routes
            plt.plot([data[data[:,0]==i][0][2], data[data[:,0]==better[loc+1]][0][2]],\
            [data[data[:,0]==i][0][3], data[data[:,0]==better[loc+1]][0][3]]\
                , c='k', linewidth=0.9, alpha=0.7)
        else:
            plt.plot([data[data[:,0]==i][0][2], data[0][2]],\
            [data[data[:,0]==i][0][3], data[0][3]],  c='k', linewidth=0.9, alpha=0.7)

    # plt.axis('equal') 
    plt.title('Plot of solution')   
    plt.xlabel('x coordinates')
    plt.ylabel('y coordinates')
    plt.show()

if __name__ == "__main__":
    plot_sol()