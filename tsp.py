""" This code is written to solve the vehicle routing problem.
It is inspired by the work at: https://goo.gl/emWw54 and implements 
the algorithm presented in the paper: (Baker, Barrie M., and M. A. Ayechew. 
"A genetic algorithm for the vehicle routing problem." 
Computers & Operations Research 30.5 (2003): 787-800.)"""
"""

This Python code is based on Java code by Lee Jacobson found in an article
entitled "Applying a genetic algorithm to the travelling salesman problem"
that can be found at: http://goo.gl/cJEY1
"""

import math
import random

# Declaring the Vehicle class:     
class Vehicle:
   def __init__(self, number=None):
      self.number = None
      if number is not None:
          self.number = number
      else:
          self.number = 2  
   def getNumber(self):
      return self.number
                
# Declaring the Customer class:     
class Customer:
   def __init__(self, x=None, y=None): # class constructor
      self.x = None
      self.y = None
      if x is not None:
         self.x = x
      else:
         self.x = int(random.random() * 200)
      if y is not None:
         self.y = y
      else:
         self.y = int(random.random() * 200)
   
   def getX(self):
      return self.x
   
   def getY(self):
      return self.y
   
   def distanceTo(self, customer): 
      xDistance = abs(self.getX() - customer.getX())
      yDistance = abs(self.getY() - customer.getY())
      distance = math.sqrt( (xDistance*xDistance) + (yDistance*yDistance) )
      return distance
   
   def __repr__(self): # Returns a string version of the object
      return str(self.getX()) + ", " + str(self.getY())

# Class for Customer data handling:
class RouteManager:
   destinationCustomers = []

      self.destinationCustomers = []
   def addCustomer(self, customer):
      self.destinationCustomers.append(customer)
   
   def getCustomer(self, index):
      return self.destinationCustomers[index]
   
   def numberOfCustomers(self):
      return len(self.destinationCustomers)

# Declaring the Route class:
class Route:
   def __init__(self, routemanager, route=None):
      self.routemanager = routemanager
      self.route = []
      self.fitness = 0.0
      self.distance = 0
      if route is not None:
         self.route = route
      else:
         for i in range(0, self.routemanager.numberOfCustomers()):
            self.route.append(None)
   
   def __len__(self):
      return len(self.route)
   
   def __getitem__(self, index):
      return self.route[index]
   
   def __setitem__(self, key, value):
      self.route[key] = value
   
   def __repr__(self):
      geneString = "|"
      for i in range(0, self.routeSize()):
         geneString += str(self.getCustomer(i)) + "|"
      return geneString
   
   def generateIndividual(self):
      # Loop through all vehicles and randomly assign them to customers
      for customerIndex in range(0, self.routemanager.numberOfCustomers()):
         self.setCustomer(customerIndex, random.randint(1, vehicles.number))   
   
   def getCustomer(self, routePosition):
      # return self.route[routePosition]
      return RouteManager.destinationCustomers[routePosition]
   
   def setCustomer(self, routePosition, customer):
      self.route[routePosition] = customer
      self.fitness = 0.0
      self.distance = 0
   
   def getFitness(self):
      if self.fitness == 0:
         self.fitness = 1/float(self.getDistance())
      return self.fitness
   
   def getDistance(self):
      if self.distance == 0:
         routeDistance = 0
         for customerIndex in range(0, self.routeSize()):
            fromCustomer = self.getCustomer(customerIndex)
            destinationCustomer = None
            # Check we're not on our tour's last city, if we are set our 
            # tour's final destination city to our starting city
            if customerIndex+1 < self.routeSize():
               destinationCustomer = self.getCustomer(customerIndex+1)
            else:
               destinationCustomer = self.getCustomer(0)
            routeDistance += fromCustomer.distanceTo(destinationCustomer)
         self.distance = routeDistance
      return self.distance
   
   def routeSize(self):
      return len(self.route)
   
   def containsCustomer(self, customer):
      return customer in self.route

# Declaring the Population class:
class Population:
   def __init__(self, routemanager, populationSize, initialise):
      self.routes = []
      for i in range(0, populationSize):
         self.routes.append(None)
      
      if initialise:
         for i in range(0, populationSize):
    #    Modifications here >>>> Marwan
            newRoute = Route(routemanager)
            newRoute.generateIndividual()
            self.saveRoute(i, newRoute)
      
   def __setitem__(self, key, value):
      self.routes[key] = value
   
   def __getitem__(self, index):
      return self.routes[index]

   def saveRoute(self, index, route):
      self.routes[index] = route # route.route
   
   def getRoute(self, index):
      return self.routes[index]
   
   def getFittest(self):
      fittest = self.routes[0]
      for i in range(0, self.populationSize()):
         if fittest.getFitness() <= self.getRoute(i).getFitness():
            fittest = self.getRoute(i)
      return fittest
   
   def populationSize(self):
      return len(self.routes)

# Declaring GA class:
class GA:
   def __init__(self, routemanager):
      self.routemanager = routemanager
      self.mutationRate = 0.015
      self.tournamentSize = 5
      self.elitism = True
   
   def evolvePopulation(self, pop):
      newPopulation = Population(self.routemanager, pop.populationSize(), False)
      elitismOffset = 0
      if self.elitism:
         newPopulation.saveRoute(0, pop.getFittest())
         elitismOffset = 1
      
      for i in range(elitismOffset, newPopulation.populationSize()):
         parent1 = self.tournamentSelection(pop)
         parent2 = self.tournamentSelection(pop)
         child = self.crossover(parent1, parent2)
         newPopulation.saveRoute(i, child)
      
      for i in range(elitismOffset, newPopulation.populationSize()):
         self.mutate(newPopulation.getRoute(i))
      
      return newPopulation
   
   def crossover(self, parent1, parent2):
      child = Route(self.routemanager)
      
      startPos = int(random.random() * parent1.routeSize())
      endPos = int(random.random() * parent1.routeSize())
      
      for i in range(0, child.routeSize()):
         if startPos < endPos and i > startPos and i < endPos:
            child.setCustomer(i, parent1.getCustomer(i))
         elif startPos > endPos:
            if not (i < startPos and i > endPos):
               child.setCustomer(i, parent1.getCustomer(i))
      
      for i in range(0, parent2.routeSize()):
         if not child.containsCustomer(parent2.getCustomer(i)):
            for ii in range(0, child.routeSize()):
               if child.getCustomer(ii) == None:
                  child.setCustomer(ii, parent2.getCustomer(i))
                  break
      
      return child
   
   # Mutate a route using 2-point swap mutation:
   def mutate(self, route):
      for routePos1 in range(0, route.routeSize()):
         if random.random() < self.mutationRate:
            routePos2 = int(route.routeSize() * random.random())
            
            customer1 = route.getCustomer(routePos1)
            customer2 = route.getCustomer(routePos2)
            
            route.setCustomer(routePos2, customer1)
            route.setCustomer(routePos1, customer2)

   # Select candidate route for crossover:
   def tournamentSelection(self, pop):
      tournament = Population(self.routemanager, self.tournamentSize, False)
      for i in range(0, self.tournamentSize):
         randomId = int(random.random() * pop.populationSize())
         tournament.saveRoute(i, pop.getRoute(randomId))
      fittest = tournament.getFittest()
      return fittest

if __name__ == '__main__':
   
   routemanager = RouteManager()
   vehicles = Vehicle(4)
   
   # Create and add our customers
   customer1 = Customer(60, 200)
   routemanager.addCustomer(customer1)
   customer2 = Customer(180, 200)
   routemanager.addCustomer(customer2)
   customer3 = Customer(80, 180)
   routemanager.addCustomer(customer3)
   customer4 = Customer(140, 180)
   routemanager.addCustomer(customer4)
   customer5 = Customer(20, 160)
   routemanager.addCustomer(customer5)
   customer6 = Customer(100, 160)
   routemanager.addCustomer(customer6)
   customer7 = Customer(200, 160)
   routemanager.addCustomer(customer7)
   customer8 = Customer(140, 140)
   routemanager.addCustomer(customer8)
   customer9 = Customer(40, 120)
   routemanager.addCustomer(customer9)
   customer10 = Customer(100, 120)
   routemanager.addCustomer(customer10)
   customer11 = Customer(180, 100)
   routemanager.addCustomer(customer11)
   customer12 = Customer(60, 80)
   routemanager.addCustomer(customer12)
   customer13 = Customer(120, 80)
   routemanager.addCustomer(customer13)
   customer14 = Customer(180, 60)
   routemanager.addCustomer(customer14)
   customer15 = Customer(20, 40)
   routemanager.addCustomer(customer15)
   customer16 = Customer(100, 40)
   routemanager.addCustomer(customer16)
   customer17 = Customer(200, 40)
   routemanager.addCustomer(customer17)
   customer18 = Customer(20, 20)
   routemanager.addCustomer(customer18)
   customer19 = Customer(60, 20)
   routemanager.addCustomer(customer19)
   customer20 = Customer(160, 20)
   routemanager.addCustomer(customer20)
   
   # Initialize population with 50 individuals
   pop = Population(routemanager, 50, True)
   print("Initial distance: " + str(pop.getFittest().getDistance()))
   
   # Evolve population for 500 generations
   ga = GA(routemanager)
   pop = ga.evolvePopulation(pop)
   for i in range(0, 500):
      pop = ga.evolvePopulation(pop)
   
   # Print final results
   print("Finished")
   print ("Final distance: " + str(pop.getFittest().getDistance()))
   print ("Solution:")
   print (pop.getFittest())