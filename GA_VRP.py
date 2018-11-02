# This code is intended to solve the vehicle routing problem using GAFT Python library
# The code implements the algorithm presented in the paper:
    # (Baker, Barrie M., and M. A. Ayechew. 
    # "A genetic algorithm for the vehicle routing problem." 
    # Computers & Operations Research 30.5 (2003): 787-800.)

from gaft.components import DecimalIndividual
from gaft.components import Population

from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation

from gaft.analysis import ConsoleOutput # To output to the console window
from gaft import GAEngine

from enum import Enum

"""from gaft.operators import two-pointCrossover""" # We need to add this!

# Declaring distances between 20 customers:
dist = [10, 15, 12, 11, 14, 5, 3, 20, 18, 22, 7, 10, 15, 18, 5, 3, 7, 16, 10, 11]

# Create Vehicle enumerator:
class Vehicle(Enum):
    A = 1
    B = 2
    C = 3
    D = 4

# Create decimal individual object:

indv = DecimalIndividual(ranges=[(1,4),(1,4),(1,4),(1,4),(1,4),(1,4),(1,4)
                        ,(1,4),(1,4),(1,4),(1,4),(1,4),(1,4),(1,4),(1,4),(1,4)
                        ,(1,4),(1,4),(1,4),(1,4)], eps=0.001)

# indv = DecimalIndividual(ranges=[Vehicle, Vehicle, Vehicle, Vehicle, Vehicle, Vehicle
#                         , Vehicle, Vehicle, Vehicle, Vehicle, Vehicle, Vehicle, Vehicle
#                         , Vehicle, Vehicle, Vehicle, Vehicle, Vehicle, Vehicle, Vehicle], eps=0.001)


# Create population object, initialize with 50 individuals:
population = Population(indv_template=indv, size=50).init()

# Create genetic operators: selection, crossover, mutation:
selection = TournamentSelection()

uniform_crossover = UniformCrossover(pc=0.8, pe=0.5)
"""two_pcrossover = Two_PointCrossover(pc=0.8, pe=0.5)""" # We need to add this

mutation = FlipBitMutation(pm=0.1)

# Create GA object:
engine = GAEngine(population=population, selection=selection, crossover=uniform_crossover,
                  mutation=mutation, analysis=[ConsoleOutput])

# Define the fitness function:
from math import sin, cos

@engine.fitness_register
@engine.minimize
def fitness(indv):
    x = indv.solution
    return sum(x)

# Run the engine for 500 generations:
engine.run(ng=500)