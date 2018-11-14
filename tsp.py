""" This code is written to solve the vehicle routing problem.
Inspired by the work at: https://goo.gl/emWw54, it implements 
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
class CustomerManager:
   def __init__(self):
      self.destinationCustomers = []

   def addCustomer(self, customer):
      self.destinationCustomers.append(customer)
   
   def getCustomer(self, index):
      return self.destinationCustomers[index]
   
   def numberOfCustomers(self):
      return len(self.destinationCustomers)

# Class for Route data handling:
class RouteManager:
   def __init__(self):
      self.routes = []

   def addRoute(self, route):
      self.routes.append(route)
   
   def getRoute(self, index):
      return self.routes[index]
   
   def numberOfRoutes(self):
      return len(self.routes)

# Declaring the Route class:
class Route:
   def __init__(self, routemanager, customermanager, route=None):
      self.routemanager = routemanager
      self.customermanager = customermanager
      self.route = []
      self.fitness = 0.0
      self.distance = 0
      if route is not None:
         self.route = route
      else:
         for i in range(0, customermanager.numberOfCustomers()):
            self.route.append(None)
   
   def __len__(self):                   # Get the length of the route chromosome
      return len(self.route)
   
   def __getitem__(self, index):        # Get an item in the route chromosome
      return self.route[index]
   
   def __setitem__(self, key, value):   # Set an item in the route chromosome
      self.route[key] = value
   
   def __repr__(self):                  # Show the route chromosome
      geneString = "|"
      for i in range(0, self.routeSize()):
         geneString += str(self.routemanager.getRoute(i)) + "|"
      return geneString
   
   def generateIndividual(self, vehicles):        # Generate a route chromosome
      # Loop through all vehicles and randomly assign them to customers
        
      for routeIndex in range(0, self.customermanager.numberOfCustomers()): 
         self.setRoute(routeIndex, random.randint(1, vehicles.number))   
   
   def getCustomer(self, routePosition):# Get a customer 
      return self.customermanager.destinationCustomers[routePosition]
 
   def getRoute(self, routePosition):   # Get a route
      return self.routemanager.routes[routePosition]
    
   def setRoute(self, routePosition, route):
      self.route[routePosition] = route
      self.fitness = 0.0
      self.distance = 0

   def getRouteGene(self, routePosition, genePosition):   # Get a route gene
      return self.routemanager.routes[routePosition][genePosition]
    
   def setRouteGene(self, genePosition, route):
      self.route[genePosition] = route[genePosition]
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
   
   def containsRoute(self, route):
      return route in self.route

# Declaring the Population class:
class Population:
   def __init__(self, routemanager, customermanager,
                vehicles, populationSize, initialize):
      self.routes = []
      for i in range(0, populationSize):
         self.routes.append(None)
         
      if initialize:
         for index in range(0, populationSize):
            newRoute = Route(routemanager, customermanager)
            newRoute.generateIndividual(vehicles)
            self.saveRoute(index, newRoute)
            self.updateRoutes(index)
      
   def __setitem__(self, key, value):
      self.routes[key] = value
   
   def __getitem__(self, index):
      return self.routes[index]

   def saveRoute(self, index, route):
      self.routes[index] = route
   
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

   def updateRoutes(self, index):
      routemanager.addRoute(self.routes[index].route)

# Declaring GA class:
class GA:
   def __init__(self, routemanager, customermanager, vehicles):
      self.routemanager = routemanager
      self.customermanager = customermanager
      self.mutationRate = 0.015
      self.tournamentSize = 5
      self.elitism = True
   
   def evolvePopulation(self, vehicles, pop):
      newPopulation = Population(self.routemanager, self.customermanager,
                                 vehicles, pop.populationSize(), False)
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
      
      for index in range(newPopulation.populationSize()):
         newPopulation.updateRoutes(index)
      return newPopulation
   
   def crossover(self, parent1, parent2):
      child = Route(self.routemanager, self.customermanager)
      
      startPos = int(random.random() * parent1.routeSize())
      endPos = int(random.random() * parent1.routeSize())
          
      # return child
      for i in range(0, parent1.routeSize()):
         if startPos <= endPos and i >= startPos and i <= endPos:
            # child.setRoute(i, parent1.getRoute(i))
            child.setRouteGene(i, parent1.route)
         elif startPos > endPos:
            if not (i < startPos and i > endPos):
            #    child.setRoute(i, parent1.getRoute(i))
               child.setRouteGene(i, parent1.route)
      
      for ii in range(0, parent2.routeSize()):
         if child.route[ii] == None:
            # child.setRoute(ii, parent2.getRoute(i))
            child.setRouteGene(ii, parent2.route)
            # break    
      return child
   
   # Mutate a route using 2-point swap mutation:
   def mutate(self, route):
      for routePos1 in range(0, route.routeSize()):
         if random.random() < self.mutationRate:
            routePos2 = int(route.routeSize() * random.random())
            
            route1 = route[routePos1]
            route2 = route[routePos2]
            
            route[routePos1] = route2
            route[routePos2] = route1

   # Select candidate route for crossover:
   def tournamentSelection(self, pop):
      tournament = Population(self.routemanager, self.customermanager, vehicles, 
                              self.tournamentSize, False)
      for i in range(0, self.tournamentSize):
         randomId = int(random.random() * pop.populationSize())
         tournament.saveRoute(i, pop.getRoute(randomId))
      fittest = tournament.getFittest()
      return fittest

if __name__ == '__main__':

      routemanager = RouteManager()
      customermanager = CustomerManager()
      vehicles = Vehicle(4)

      # Create and add our customers
      customer1 = Customer(60, 200)
      customermanager.addCustomer(customer1)
      customer2 = Customer(180, 200)
      customermanager.addCustomer(customer2)
      customer3 = Customer(80, 180)
      customermanager.addCustomer(customer3)
      customer4 = Customer(140, 180)
      customermanager.addCustomer(customer4)
      customer5 = Customer(20, 160)
      customermanager.addCustomer(customer5)
      customer6 = Customer(100, 160)
      customermanager.addCustomer(customer6)
      customer7 = Customer(200, 160)
      customermanager.addCustomer(customer7)
      # customer8 = Customer(140, 140)
      # customermanager.addCustomer(customer8)
      # customer9 = Customer(40, 120)
      # customermanager.addCustomer(customer9)
      # customer10 = Customer(100, 120)
      # customermanager.addCustomer(customer10)
      # customer11 = Customer(180, 100)
      # customermanager.addCustomer(customer11)
      # customer12 = Customer(60, 80)
      # customermanager.addCustomer(customer12)
      # customer13 = Customer(120, 80)
      # customermanager.addCustomer(customer13)
      # customer14 = Customer(180, 60)
      # customermanager.addCustomer(customer14)
      # customer15 = Customer(20, 40)
      # customermanager.addCustomer(customer15)
      # customer16 = Customer(100, 40)
      # customermanager.addCustomer(customer16)
      # customer17 = Customer(200, 40)
      # customermanager.addCustomer(customer17)
      # customer18 = Customer(20, 20)
      # customermanager.addCustomer(customer18)
      # customer19 = Customer(60, 20)
      # customermanager.addCustomer(customer19)
      # customer20 = Customer(160, 20)
      # customermanager.addCustomer(customer20)

      # Initialize population with 50 individuals
      pop = Population(routemanager, customermanager, vehicles, 5, True)
      print("Initial distance: " + str(pop.getFittest().getDistance()))

      # Evolve population for 500 generations
      ga = GA(routemanager, customermanager, vehicles)
      pop = ga.evolvePopulation(vehicles, pop)
      for i in range(0, 500):
            pop = ga.evolvePopulation(vehicles, pop)

      # Print final results
      print("Finished")
      print ("Final distance: " + str(pop.getFittest().getDistance()))
      print ("Solution:")
      print (pop.getFittest())