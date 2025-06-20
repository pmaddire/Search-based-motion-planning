from pqdict import pqdict
import math
import numpy as np
import math
from scipy.spatial.distance import cdist

class AStarNode(object):
  def __init__(self, pqkey, coord, hval):
    self.pqkey = pqkey
    self.coord = coord
    self.g = math.inf
    self.h = hval
    self.parent_node = None
    self.parent_action = None
    self.closed = False
    self.open = False
  def __lt__(self, other):
    return self.g < other.g

class Enviroment:
   def __init__ (self, goal, boundary, blocks, precision):
      self.goal = goal
      self.boundary = boundary
      self.blocks = blocks 
      self.precision = precision
   def isGoal(self, node):
      #return (tuple(node) == tuple(self.goal))
      if (np.linalg.norm(np.array(node) - np.array(self.goal)) < (self.precision * 0.8)):
         if not self.CollosionCheck(node, self.goal, self.boundary, self.blocks):
            return True
      return False

   def getSuccessors(self, node):
      x = node.coord[0]
      y = node.coord[1]
      z = node.coord[2]

      successor_list = []
      cost_list = []
      action_list = []
      direc = [-self.precision, 0.0, self.precision] 
      for dx in direc:
         #print("iter")
         for dy in direc:
            for dz in direc:
               if dx == dy == dz ==0:
                  continue 
               neighbor_coords = (x + dx, y+ dy, z + dz)
               #print(neighbor_coords)
               if not self.is_valid( neighbor_coords, node):
                  #print("valid")
                  if dx != 0 and dy !=0 and dz != 0:
                     cost = np.sqrt(3) 
                  elif (dx !=0 and dy !=0) or (dx !=0 and dz!=0) or (dy !=0 and dz!=0):
                     cost = np.sqrt(2) 
                  else:
                     cost = 1.0 
                  successor_list.append(neighbor_coords)
                  cost_list.append(cost)
                  action_list.append((dx, dy, dz))
      return successor_list, cost_list, action_list

   def getHeuristic(self, node):
        return np.linalg.norm(np.array(node) - np.array(self.goal))
   
   def is_valid(self, end_node, current_node):
      #print(CollosionCheck(current_node, end_node, self.boundary, self.blocks))
      return self.CollosionCheck(current_node.coord, end_node, self.boundary, self.blocks)
   
   def CollosionCheck(self, start, end, boundary, blocks):
    num_points = 10
    x_points =  np.linspace(start[0], end[0], num_points)
    y_points =  np.linspace(start[1], end[1], num_points)
    z_points =  np.linspace(start[2], end[2], num_points)
    
    # boundry check 
    # x coord check 
    #print(x_points)
    if np.any(x_points >= boundary[0, 3]) or np.any(x_points <= boundary[0, 0]):
        #print("x out of bounds")
        #print(boundary[0, 0])
        #print(x_points)
        return True
        
    # y check 
    if np.any(y_points >= boundary[0, 4]) or np.any(y_points <= boundary[0, 1]):
        #print("y out of bounds")
        return True
        
    #z check
    if np.any(z_points >= boundary[0, 5]) or np.any(z_points <= boundary[0, 2]):
        #print("z out of bounds")
        return True 
         

    # Rectangualr objects check
    #print(blocks.shape[0])
    for i in range(blocks.shape[0]):
        block = blocks[i,:]
        min_x = block[0]
        min_y = block[1]
        min_z = block[2]
        max_x = block[3]
        max_y = block[4]
        max_z = block[5]

        for j in range(num_points):
          if ((min_x <= x_points[j]) & (x_points[j] <= max_x)) & ((min_y <= y_points[j]) & (y_points[j] <= max_y)) & ((min_z <= z_points[j]) & (z_points[j] <= max_z)):
            #print(block)
            #print(x_points[j], y_points[j], z_points[j])
            #print("object collision")
            return True            

    return False
      
class Astar:

    def __init__(self, boundary, blocks, start, goal, precision,epsilon, reopen_nodes):
        self.boundary = boundary
        self.blocks = blocks
        self.start = start
        self.goal = goal
        self.epsilon = epsilon 
        self.precision = precision
        self.reopen_nodes = reopen_nodes
        #self.planning()

    def planning(self):
        Open = pqdict() 
        Graph = {}
        env = Enviroment(self.goal, self.boundary, self.blocks, self.precision)
         # current node
        curr = AStarNode(tuple(self.start), self.start, env.getHeuristic(self.start))
        curr.g = 0
        Graph[curr.pqkey] = curr

        while True:
           if env.isGoal(curr.coord):
              return self.recoverPath(curr), len(Graph)
           
           curr.closed = True 
           self.updateData(curr, Graph, Open, env)

           if not Open: 
              return self.recoverPath(curr), len(Graph)
           
           curr = Open.popitem()[1][1]
           #print("current expansion:", curr.coord)


        #g = math.inf
        #start_coords = tuple(self.start)
        #Open.additem((start_coords),math.inf + self.epsilon*self.heuristic(self.start))
        #print(Open)

    def recoverPath(self, curr):
       path = []
       curr_node = curr

       while curr_node is not None:
          path.append(curr_node.coord)
          #print(curr_node.coord)
          curr_node = curr_node.parent_node

       path.reverse()
   
       if not np.array_equal(path[-1],self.goal):
          path.append(self.goal)
       #print(path)
       return np.array(path)

    def updateData(self, curr, Graph, Open, env):
         successor_list, cost_list, action_list = env.getSuccessors(curr)
         for s_coord, s_cost, s_action in zip(successor_list, cost_list, action_list):
            s_key = tuple(s_coord)
            if s_key not in Graph:
               Graph[s_key] = AStarNode(s_key, s_coord, env.getHeuristic(s_coord))
               Graph[s_key].h = env.getHeuristic(s_coord)
            child = Graph[s_key]

            tentative_g = curr.g + s_cost
            if(tentative_g < child.g):
               child.parent_node, child.parent_action = curr, s_action
               #print(f"Setting parent of {child.coord} to { child.parent_node}")
               child.g = tentative_g

               fval = tentative_g + self.epsilon*child.h
               if child.open:
                  Open[s_key] = (fval, child)
                  Open.heapify(s_key)
               elif child.closed and self.reopen_nodes:
                  Open[s_key] = (fval, child)
                  child.open, child.closed = True, False
               else:
                  Open[s_key] = (fval, child)
                  child.open = True 

    def heuristic(self, current):
        dist_vector = self.goal - current
        h = np.linalg.norm(dist_vector) #L2 eucliedian norm 
        #h = cdist(self.goal, current, metric='cityblock') #Manhattan distance
        #h = np.max(np.abs(self.goal - current)) # Chebyshev_distance 
        #h = self.octile_distance(self.goal, current)

        return h
    def octile_distance(self, a, b):
       dx = np.abs(a[0] - b[0])
       dy = np.abs(a[1] - b[1])
       return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)