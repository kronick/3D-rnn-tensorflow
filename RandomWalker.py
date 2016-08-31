import pyassimp
import random
import numpy as np
import math

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
    self.unique_vertex_indices = {}
    self.unique_vertices = []
    self.vertex_connections = {} # Should end up with same number of elements as self.mesh.vertices
    
    _face_normals = {}    # Indexed by vertex
    self.vertex_normals = {}  # Average of face normals

    def match_unique_vertex(p):
      plist = tuple(p.tolist())

      if plist in self.unique_vertex_indices:
        return self.unique_vertex_indices[plist]

      # No match found; add this point
      self.unique_vertices.append(p)
      idx = len(self.unique_vertices) - 1
      self.unique_vertex_indices[plist] = idx
      _face_normals[idx] = []
      return idx

    for a, b, c in self.mesh.faces:
      # Get index for each point in the unique_vertices list
      a_coords = self.mesh.vertices[a]
      b_coords = self.mesh.vertices[b]
      c_coords = self.mesh.vertices[c]

      a_index = match_unique_vertex(a_coords)
      b_index = match_unique_vertex(b_coords)
      c_index = match_unique_vertex(c_coords)
           

      if not self.vertex_connections.has_key(a_index):
        self.vertex_connections[a_index] = set()
      if not self.vertex_connections.has_key(b_index):
        self.vertex_connections[b_index] = set()
      if not self.vertex_connections.has_key(c_index):
        self.vertex_connections[c_index] = set()

      self.vertex_connections[a_index].add(b_index)
      self.vertex_connections[a_index].add(c_index)

      self.vertex_connections[b_index].add(a_index)
      self.vertex_connections[b_index].add(c_index)

      self.vertex_connections[c_index].add(b_index)
      self.vertex_connections[c_index].add(a_index)

      # Calculate face normal and append to each point
      I = b_coords - a_coords # Get vectors on the face
      J = c_coords - a_coords
      cross = np.cross(I, J)  # Cross product gives vector perpendicular to face
      mag = np.sqrt(cross.dot(cross)) # Calculate magnitude
      norm = cross / mag # Normalize the cross product perpendicular vector to get face normal
  
      _face_normals[a_index].append(norm)
      _face_normals[b_index].append(norm)
      _face_normals[c_index].append(norm)

    # Average face normals to get vertex normals
    for idx, norms in _face_normals.iteritems():
      S = np.sum(norms, 0)
      avg = S / len(norms)
      mag = np.sqrt(avg.dot(avg))
      avg /= mag
      self.vertex_normals[idx] = avg

    self.loaded = True

  def get_smoothest_index(self, curr_point, prev_point, connections):
    A = curr_point - prev_point
    
    A = np.array([A[0], A[1], A[2]])
    mag_A = math.sqrt(A[0]*A[0] + A[1]*A[1] + A[2]*A[2])
    A /= mag_A


    max_angle = 0
    max_idx = 0
    max_dot = 0
    n = -1
    for i, c in enumerate(connections):
      p = self.unique_vertices[c]
      B = curr_point - p
      mag_B = math.sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2])
      B /= mag_B

      dot = (B[0] * A[0] + B[1] * A[1] + B[2] * A[2])
      dot = min(dot, 1)
      dot = max(dot, -1)
      
      angle = math.acos(dot)
      if angle > max_angle:

        max_angle = angle
        max_idx = c
        n = i

    return max_idx

  def walk(self, n_steps, relative = True, smooth = 0):
    """ Randomly walk from point to point on the mesh, starting at a random point """
    if not self.loaded:
       return []

    smooth = min(smooth, 1.0)

    points_out = np.zeros((n_steps, 4))
    normals_out = np.zeros((n_steps, 3))

    # Start at a random point
    previous_vertex_index = -1
    last_vertex_index = random.randint(0, len(self.unique_vertices)-1)
    p = self.unique_vertices[last_vertex_index]
    last_point = np.array([p[0],p[1],p[2], 0])
    points_out[0] = np.array([0,0,0,0]) if relative else last_point
    normals_out[0] = self.vertex_normals[last_vertex_index]

    for i in xrange(n_steps-1):
      # Get vertex connections
      connections = self.vertex_connections.get(last_vertex_index)
      if connections == None or len(connections) == 0:
        break

      # Choose a random next connected vertex
      if random.random() > 1.0 - smooth:
        next_vertex_index = self.get_smoothest_index(self.unique_vertices[last_vertex_index], self.unique_vertices[previous_vertex_index], connections - set((previous_vertex_index,)))
      else:
        next_vertex_index = random.sample(connections - set((previous_vertex_index,)), 1)[0]
      #print "At vertex #{} - connected to ({}) - going to #{}".format(last_vertex_index, connections, next_vertex_index)
      p = self.unique_vertices[next_vertex_index]
      eos = 0 if (i < n_steps - 2) else 1
      next_point = np.array([p[0],p[1],p[2], eos])
      # Add either its relative or absolute position to the list
      points_out[i+1] = next_point - last_point if relative else next_point
      normals_out[i+1] = self.vertex_normals[next_vertex_index]

      previous_vertex_index = last_vertex_index
      last_vertex_index = next_vertex_index
      last_point = next_point
    
    #return np.array(points_out)
    return np.concatenate((points_out,normals_out), axis=1) # Return array with points and normals concatenated -- shape is (n_steps, 7)

if __name__ == "__main__":
  w = RandomWalker("rock.obj")
  points = w.walk(100)
  for p in points:
    print p