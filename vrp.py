# encoding: utf-8
import sys
import random
import math
import matplotlib.pyplot as plt

vrp = {}

## First reading the VRP from the input ##

def readinput():
	try:
		line = input().strip()
		while line == '' or line.startswith('#'):
			line = input().strip()
		return line
	except EOFError:
		return None

line = readinput()

if line == None:
	print(sys.stderr, 'Empty input!')
	exit(1)

if line.lower() != 'params:':
	print(sys.stderr, 'Invalid input: it must be the VRP initial params at first!')
	exit(1)

line = readinput()
if line == None:
	print(sys.stderr, 'Invalid input: missing VRP inital params and nodes!')
	exit(1)
while line.lower() != 'nodes:':
	inputs = line.split()
	if len(inputs) < 2:
		print(sys.stderr, 'Invalid input: too few arguments for a param!')
		exit(1)
	if inputs[0].lower() == 'capacity':
		vrp['capacity'] = float(inputs[1])
		# Validating positive non-zero capacity
		if vrp['capacity'] <= 0:
			print(sys.stderr, 'Invalid input: capacity must be neither negative nor zero!')
			exit(1)
	else:
		print >> sys.stderr, 'Invalid input: invalid VRP initial param!'
		exit(1)
	line = readinput()
	if line == None:
		print(sys.stderr, 'Invalid input: missing nodes!')
		exit(1)

if not set(vrp).issuperset({'capacity'}):
	print(sys.stderr, 'Invalid input: missing some required VRP initial params!')
	exit(1)

line = readinput()
vrp['nodes'] = [{'label' : 'depot', 'demand' : 0, 'posX' : 0, 'posY' : 0}]

# for i in range(0,7): #TEMPORARY FOR TESTING PURPOSES
while line != None:
	inputs = line.split()
	if len(inputs) < 4:
		print(sys.stderr, 'Invalid input: too few arguments for a node!')
		exit(1)
	node = {'label' : inputs[0], 'demand' : float(inputs[1]), 'posX' : float(inputs[2]), 'posY' : float(inputs[3])}
	# Validating demand neither negative nor zero
	if node['demand'] <= 0:
		print(sys.stderr, 'Invalid input: the demand if the node %s is negative or zero!' % node['label'])
		exit(1)
	# Validating demand not greater than capacity
	if node['demand'] > vrp['capacity']:
		print(sys.stderr, 'Invalid input: the demand of the node %s is greater than the vehicle capacity!' % node['label'])
		exit(1)
	vrp['nodes'].append(node)
	line = readinput()

# Validating no such nodes
if len(vrp['nodes']) == 0:
	print(sys.stderr, 'Invalid input: no such nodes!')
	exit(1)

## After inputting and validating it, now computing the algorithm ##

def distance(city1, city2):
	dx = city2['posX'] - city1['posX']
	dy = city2['posY'] - city1['posY']
	return math.sqrt(dx * dx + dy * dy)

def fitness(individual):
	# The first distance is from depot to the first node of the first route
	totaldist = distance(vrp['nodes'][0], vrp['nodes'][individual[0]])
	# Then calculating the distances between the nodes
	for i in range(len(individual) - 1):
		prev = vrp['nodes'][individual[i]]
		next = vrp['nodes'][individual[i + 1]]
		totaldist += distance(prev, next)
	# The last distance is from the last node of the last route to the depot
	totaldist += distance(vrp['nodes'][individual[len(individual) - 1]], vrp['nodes'][0])
	return 1/float(totaldist)

def adjust(individual):
	# Adjust repeated
	repeated = True
	while repeated:
		repeated = False
		for i1 in range(len(individual)):
			for i2 in range(i1):
				if individual[i1] == individual[i2]:
					haveAll = True
					for nodeId in range(len(vrp['nodes'])):
						if nodeId not in individual:
							individual[i1] = nodeId
							haveAll = False
							break
					if haveAll:
						del individual[i1]
					repeated = True
				if repeated: break
			if repeated: break
	# Adjust capacity exceed
	i = 0				  # index
	reqcap = 0.0				  # required capacity
	cap = vrp['capacity'] # available vehicle capacity
	while i < len(individual):
		reqcap += vrp['nodes'][individual[i]]['demand']
		if reqcap > cap:
			individual.insert(i, 0)
			reqcap = 0.0
		i += 1
	i = len(individual) - 2
	# Adjust two consective depots
	while i >= 0:
		if individual[i] == 0 and individual[i + 1] == 0:
			del individual[i]
		i -= 1


popsize = int(sys.argv[1])
iterations = int(sys.argv[2])

# popsize = 50      #TEMPORARY FOR TESTING PURPOSES
# iterations = 100  #TEMPORARY FOR TESTING PURPOSES

pop = []

# Generating random initial population
for i in range(popsize):
	individual = list(range(1, len(vrp['nodes'])))
	random.shuffle(individual)
	pop.append(individual)
for individual in pop:
	adjust(individual)

# Running the genetic algorithm
for i in range(iterations):
	nextPop = []

	# Each one of this iteration will generate two descendants individuals. 
	# Therefore, to guarantee same population size, this will iterate half population size times:

	for j in range(int(len(pop) / 2)):
		# Selecting randomly 4 individuals to select 2 parents by a binary tournament
		parentIds = set()
		while len(parentIds) < 4:
			parentIds |= {random.randint(0, len(pop) - 1)}
		parentIds = list(parentIds)
		# Selecting 2 parents with the binary tournament
		parent1 = pop[parentIds[0]] if fitness(pop[parentIds[0]]) > fitness(pop[parentIds[1]]) else pop[parentIds[1]]
		parent2 = pop[parentIds[2]] if fitness(pop[parentIds[2]]) > fitness(pop[parentIds[3]]) else pop[parentIds[3]]
		# Selecting two random cutting points for crossover, with the same points (indexes) for both parents, based on the shortest parent
		cutIdx1, cutIdx2 = random.randint(1, min(len(parent1), len(parent2)) - 1), random.randint(1, min(len(parent1), len(parent2)) - 1)
		cutIdx1, cutIdx2 = min(cutIdx1, cutIdx2), max(cutIdx1, cutIdx2)
		# Doing crossover and generating two children
		child1 = parent1[:cutIdx1] + parent2[cutIdx1:cutIdx2] + parent1[cutIdx2:]
		child2 = parent2[:cutIdx1] + parent1[cutIdx1:cutIdx2] + parent2[cutIdx2:]
		nextPop += [child1, child2]
	# Doing mutation: swapping two positions in one of the individuals, with 1:15 probability
	if random.randint(1, 15) == 1:
		ptomutate = nextPop[random.randint(0, len(nextPop) - 1)]
		i1 = random.randint(0, len(ptomutate) - 1)
		i2 = random.randint(0, len(ptomutate) - 1)
		ptomutate[i1], ptomutate[i2] = ptomutate[i2], ptomutate[i1]
	# Adjusting individuals
	for individual in nextPop:
		adjust(individual)
	# Updating population generation
	pop = nextPop

# Selecting the best individual, which is the final solution
better = None
bf = float('inf')
for individual in pop:
	f = fitness(individual)
	if f < bf:
		bf = 1/f
		better = [0]+individual


## After processing the algorithm, now outputting it ##


# Printing the solution
print (' route:')
print ('depot')
for i, nodeIdx in enumerate(better):
	print (vrp['nodes'][nodeIdx]['label'])
	if vrp['nodes'][nodeIdx]['label']=='depot':
		plt.scatter(vrp['nodes'][nodeIdx]['posX'], vrp['nodes'][nodeIdx]['posY'],None,'r','x')
	else:
		plt.scatter(vrp['nodes'][nodeIdx]['posX'], vrp['nodes'][nodeIdx]['posY'],None,'b')
		plt.annotate(vrp['nodes'][nodeIdx]['label']+',\n'+str(vrp['nodes'][nodeIdx]['demand']), 
					xy=(vrp['nodes'][nodeIdx]['posX'], vrp['nodes'][nodeIdx]['posY']))
	if i != len(better)-1:
		nextCityIdx = better[i+1]
	else:
		nextCityIdx = 0
	plt.plot([vrp['nodes'][nodeIdx]['posX'],vrp['nodes'][nextCityIdx]['posX']],
				   [vrp['nodes'][nodeIdx]['posY'], vrp['nodes'][nextCityIdx]['posY']], 'k')
print ('depot')
print (' cost:')
print ('%f' % bf)
plt.grid()
plt.show()