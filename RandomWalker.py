import pyassimp
import random
import numpy as np

class RandomWalker():
  loaded = False
  def __init__(self, filename):
    """ Load the file and index the vertex graph """
    try:
      scene = pyassimp.load(filename)
      self.mesh = scene.meshes[0]

    except Exception as e:
      print e
      return


    # Calculate the vertex graph
    self.vertex_connections = {} # Should end up with same number of elements as self.mesh.vertices
    for a, b, c in self.mesh.faces:
      if not self.vertex_connections.has_key(a):
        self.vertex_connections[a] = set()
      if not self.vertex_connections.has_key(b):
        self.vertex_connections[b] = set()
      if not self.vertex_connections.has_key(c):
        self.vertex_connections[c] = set()

      self.vertex_connections[a].add(b)
      self.vertex_connections[a].add(c)

      self.vertex_connections[b].add(a)
      self.vertex_connections[b].add(c)

      self.vertex_connections[c].add(b)
      self.vertex_connections[c].add(a)

    self.loaded = True


  def walk(self, n_steps, relative = True):
    """ Randomly walk from point to point on the mesh, starting at a random point """
    if not self.loaded:
       return []

    points_out = np.zeros((n_steps, 4))
    # Start at a random point
    last_vertex_index = random.sample(self.mesh.faces, 1)[0][0]
    p = self.mesh.vertices[last_vertex_index]
    last_point = np.array([p[0],p[1],p[2], 0])
    points_out[0] = np.array([0,0,0,0]) if relative else last_point

    for i in xrange(n_steps-1):
      # Get vertex connections
      connections = self.vertex_connections.get(last_vertex_index)
      if connections == None or len(connections) == 0:
        break

      # Choose a random next connected vertex
      next_vertex_index = random.sample(connections, 1)[0]
      p = self.mesh.vertices[next_vertex_index]
      eos = 0 if (i < n_steps - 2) else 1
      next_point = np.array([p[0],p[1],p[2], eos])
      # Add either its relative or absolute position to the list
      points_out[i+1] = next_point - last_point if relative else next_point

      last_vertex_index = next_vertex_index
      last_point = next_point
    
    return np.array(points_out)

if __name__ == "__main__":
  w = RandomWalker("/home/kronick/Desktop/meshlab samples/bunny2.ply")
  points = w.walk(10)
  print(points)
  print "Shape: {}".format(points.shape)