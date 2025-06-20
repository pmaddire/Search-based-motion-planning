import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#import Planner
import os
#import astar
from ompl import base as ob
from ompl import geometric as og
import OMPL_RRT
def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))
  

def load_map(fname):
  '''
  Loads the bounady and blocks from map file fname.
  
  boundary = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  
  blocks = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'],
            ...,
            ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  '''
  mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
                                    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
  blockIdx = mapdata['type'] == b'block'
  boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  return boundary, blocks


def draw_map(boundary, blocks, start, goal):
  '''
  Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
  '''
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  hb = draw_block_list(ax,blocks)
  hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
  hg = ax.plot(goal[0:1],goal[1:2],goal[2:],'go',markersize=7,markeredgecolor='k')  
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(boundary[0,0],boundary[0,3])
  ax.set_ylim(boundary[0,1],boundary[0,4])
  ax.set_zlim(boundary[0,2],boundary[0,5])
  return fig, ax, hb, hs, hg

def draw_block_list(ax,blocks):
  '''
  Subroutine used by draw_map() to display the environment blocks
  '''
  v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
  f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
  clr = blocks[:,6:]/255
  n = blocks.shape[0]
  d = blocks[:,3:6] - blocks[:,:3] 
  vl = np.zeros((8*n,3))
  fl = np.zeros((6*n,4),dtype='int64')
  fcl = np.zeros((6*n,3))
  for k in range(n):
    vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
    fl[k*6:(k+1)*6,:] = f + k*8
    fcl[k*6:(k+1)*6,:] = clr[k,:]
  
  if type(ax) is Poly3DCollection:
    ax.set_verts(vl[fl])
  else:
    pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
    pc.set_facecolor(fcl)
    h = ax.add_collection3d(pc)
    return h


def runtest(mapfile, start, goal, precision, verbose = True):
  '''
  This function:
   * loads the provided mapfile
   * creates a motion planner
   * plans a path from start to goal
   * checks whether the path is collision free and reaches the goal
   * computes the path length as a sum of the Euclidean norm of the path segments
  '''
  # Load a map and instantiate a motion planner
  boundary, blocks = load_map(mapfile)
  #MP = Planner.MyPlanner(boundary, blocks) # TODO: replace this with your own planner implementation
  #MP = astar.Astar(boundary, blocks, start, goal, precision, reopen_nodes = False, epsilon=2)
  MP = OMPL_RRT.OMPLPlanner(boundary, blocks) 
  # Display the environment
  if verbose:
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)

  # Call the motion planner
  t0 = tic()
  
  #path = MP.plan(start, goal)
  #path = MP.planning()
  path = MP.plan(start, goal, sampler_type='uniform')
  toc(t0,"Planning")
  
  # Plot the path
  if verbose:
    ax.plot(path[:,0],path[:,1],path[:,2],'r-')

  collision = False
  # TODO: You should verify whether the path actually intersects any of the obstacles in continuous space
  # TODO: You can implement your own algorithm or use an existing library for segment and 
  #       axis-aligned bounding box (AABB) intersection
  for i in range(len(path) -1): 
    start_point = path[i]
    end_point = path[i+1]
    if CollosionCheck(start_point, end_point, boundary, blocks):
      print(start_point)
      print(end_point)
      collision = True
      break

  goal_reached = sum((path[-1]-goal)**2) <= 0.1
  success = (not collision) and goal_reached
  pathlength = np.sum(np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1)))
  return success, pathlength


def test_single_cube(verbose = True):
  print('Running single cube test...\n') 
  # Get the absolute path of the current script
  script_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(script_dir, 'maps', 'single_cube.txt')

  start = np.array([7.0, 7.0, 5.5])
  goal = np.array([2.3, 2.3, 1.3])
  precision = 0.5
  #success, pathlength = runtest('./maps/single_cube.txt', start, goal, verbose)
  #success, pathlength = runtest(file_path, start, goal, verbose)
  success, pathlength = runtest(file_path, start, goal, precision, verbose)
  print('Success: %r'%success)
  print('Path length: %f'%pathlength)
  print('\n')
  

def test_maze(verbose = True):
  print('Running maze test...\n')
  script_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(script_dir, 'maps', 'maze.txt') 

  start = np.array([0.0, 0.0, 1.0])
  goal = np.array([12.0, 12.0, 5.0])
  precision = 0.5
  #success, pathlength = runtest('./maps/maze.txt', start, goal, verbose)
  #success, pathlength = runtest(file_path, start, goal, verbose)
  success, pathlength = runtest(file_path, start, goal, precision, verbose)
  print('Success: %r'%success)
  print('Path length: %f'%pathlength)
  print('\n')

    
def test_window(verbose = True):
  print('Running window test...\n') 
  script_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(script_dir, 'maps', 'window.txt')   
  start = np.array([6, -4.9, 2.8])
  goal = np.array([2.0, 19.5, 5.5])
  precision = 0.5
  #success, pathlength = runtest('./maps/window.txt', start, goal, verbose)
  #success, pathlength = runtest(file_path, start, goal, verbose)
  success, pathlength = runtest(file_path, start, goal, precision, verbose)
  print('Success: %r'%success)
  print('Path length: %f'%pathlength)
  print('\n')

  
def test_tower(verbose = True):
  print('Running tower test...\n') 
  script_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(script_dir, 'maps', 'tower.txt')   
  start = np.array([4.0, 2.5, 19.5])
  goal = np.array([2.5, 4.0, 0.5])
  precision = 0.4
  #success, pathlength = runtest('./maps/tower.txt', start, goal, verbose)
  #success, pathlength = runtest(file_path, start, goal, verbose)
  success, pathlength = runtest(file_path, start, goal, precision, verbose)
  print('Success: %r'%success)
  print('Path length: %f'%pathlength)
  print('\n')

     
def test_flappy_bird(verbose = True):
  print('Running flappy bird test...\n')
  script_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(script_dir, 'maps', 'flappy_bird.txt')   
  start = np.array([0.5, 4.5, 5.5])
  goal = np.array([19.5, 1.5, 1.5])
  precision = 0.5
  #success, pathlength = runtest('./maps/flappy_bird.txt', start, goal, verbose)
  #success, pathlength = runtest(file_path, start, goal, verbose)
  success, pathlength = runtest(file_path, start, goal, precision, verbose)
  print('Success: %r'%success)
  print('Path length: %f'%pathlength) 
  print('\n')

  
def test_room(verbose = True):
  print('Running room test...\n') 
  script_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(script_dir, 'maps', 'room.txt')   
  start = np.array([1.0, 5.0, 1.5])
  goal = np.array([9.0, 7.0, 1.5])
  precision = 0.5
  #success, pathlength = runtest('./maps/room.txt', start, goal, verbose)
  success, pathlength = runtest(file_path, start, goal, precision,verbose)
  print('Success: %r'%success)
  print('Path length: %f'%pathlength)
  print('\n')

def test_pillars(verbose = True):
  print('Running pillars test...\n') 
  script_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(script_dir, 'maps', 'pillars.txt')     
  start = np.array([0.5, 0.5, 0.5])
  goal = np.array([19, 19, 9])
  #success, pathlength = runtest('./maps/pillars.txt', start, goal, verbose)
  #success, pathlength = runtest(file_path, start, goal, verbose)
  precision = 0.5
  success, pathlength = runtest(file_path, start, goal, precision, verbose)
  print('Success: %r'%success)
  print('Path length: %f'%pathlength)
  print('\n')

def CollosionCheck(start, end, boundary, blocks):
    num_points = 20
    x_points =  np.linspace(start[0], end[0], num_points)
    y_points =  np.linspace(start[1], end[1], num_points)
    z_points =  np.linspace(start[2], end[2], num_points)
    
    # boundry check 
    # x coord check 
    #print(x_points)
    if np.any(x_points >= boundary[0, 3]) or np.any(x_points <= boundary[0, 0]):
        print("x out of bounds")
        #print(boundary[0, 0])
        #print(x_points)
        return True
        
    # y check 
    if np.any(y_points >= boundary[0, 4]) or np.any(y_points <= boundary[0, 1]):
        print("y out of bounds")
        return True
        
    #z check
    if np.any(z_points >= boundary[0, 5]) or np.any(z_points <= boundary[0, 2]):
        print("z out of bounds")
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
            print(block)
            print(x_points[j], y_points[j], z_points[j])
            print("object collision")
            return True            

    return False


if __name__=="__main__":
  test_single_cube()
  test_maze()
  test_flappy_bird()
  test_pillars()
  test_window()
  test_tower()
  test_room()
  plt.show(block=True)






