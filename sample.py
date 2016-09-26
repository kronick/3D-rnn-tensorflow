import numpy as np
import tensorflow as tf

import time
import os
import cPickle
import argparse

from utils import *
from model import Model
import random

import svgwrite
from IPython.display import SVG, display

# main code (not in a main function since I want to run this script in IPython as well).

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='sample',
                   help='filename of .svg file to output, without .svg')
parser.add_argument('--sample_length', type=int, default=800,
                   help='number of strokes to sample')
parser.add_argument('--scale_factor', type=int, default=1,
                   help='factor to scale down by for svg output.  smaller means bigger output')
parser.add_argument('--temperature', type=float, default=1.0,
                   help='factor to scale standard deviations by, creating bias towards more (>1.0) or less (<1.0) wild output')
parser.add_argument("--number_sequences", type=int, default=1,
                   help='number of writing sequences to generate')
parser.add_argument("--checkpoint", type=str, default='save',
                   help='Directory containing trained checkpoint.')
parser.add_argument("--prime_data", type=str, default=None,
                   help='.XYZ file of data points to prime each sequence with')
parser.add_argument("--prime_length", type=int, default=400,
                   help='Number of points to pull from prime sequence.')
parser.add_argument("--prime_scale", type=float, default=1,
                   help="Factor to scale prime model by.")
parser.add_argument("--distance_filter", type=float, default=10,
                   help="Ignore jumps between points greater than this distance.")

parser.add_argument("--bend_factor", type=float, default=1.0,
                   help="scale weights for given cell by this amount")
parser.add_argument("--bend_cell", type=int, default=-1,
                   help="skew the weights on this neural net cell")

parser.add_argument("--bend_randomly", type=int, default=0,
                   help="randomly bend this number of cells")

writestates_parser = parser.add_mutually_exclusive_group(required=False)
writestates_parser.add_argument('--states', dest='writestates', action='store_true')
writestates_parser.add_argument('--nostates', dest='writestates', action='store_false')
parser.set_defaults(writestates=True)

concatenate_parser = parser.add_mutually_exclusive_group(required=False)
concatenate_parser.add_argument('--concatenate', dest='concatenate', action='store_true')
concatenate_parser.add_argument('--separate', dest='concatenate', action='store_false')
parser.set_defaults(concatenate=True)

relative_parser = parser.add_mutually_exclusive_group(required=False)
relative_parser.add_argument('--relative', dest='relative', action='store_true')
relative_parser.add_argument('--absolute', dest='relative', action='store_false')
parser.set_defaults(relative=False)

sample_args = parser.parse_args()

with open(os.path.join(sample_args.checkpoint, 'config.pkl')) as f:
    saved_args = cPickle.load(f)

model = Model(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

ckpt = tf.train.get_checkpoint_state(sample_args.checkpoint)
print "loading model: ",ckpt.model_checkpoint_path


saver.restore(sess, ckpt.model_checkpoint_path)

if sample_args.bend_cell != -1:
  idx = sample_args.bend_cell
  factor = sample_args.bend_factor

  n_cells = saved_args.rnn_size

  layer = int(idx / n_cells)
  matrix_name = "rnnlm/MultiRNNCell/Cell" + str(layer) + "/BasicLSTMCell/Linear/Matrix:0" 
  #matrix_name = "rnnlm/MultiRNNCell/Cell" + str(layer) + "/BasicLSTMCell/Linear/Bias:0" 
  rnn_index = idx % n_cells

  for v in tf.trainable_variables():
    if v.name == matrix_name:
      print "Scaling weights for cell {}".format(v.name)
      print "index: {}".format(rnn_index)
      value = sess.run(v.value())
      print value.shape
      print value

      # Columns in the matrix are weights for each cell x 4 types of inputs to LSTM cell: input gate, new input, forget gate, output gate
      # The matrix is laid out with all weight types grouped together
      # i.e. for a network with N cells there will be N input gate weights, then N new input weights, then N forget weights, then N outputs weights
      # To change the weights on a single cell indexed by i, we have to change columns [i+0N, i+1N, i+2N, i+3N]

      for j in range(0,4):
        value[:,rnn_index + j * n_cells] *= factor

      sess.run(v.assign(value))

if sample_args.bend_randomly > 0:
  factor = sample_args.bend_factor

  n_cells = saved_args.rnn_size

  for v in tf.trainable_variables():
    if v.name == "rnnlm/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix:0":
      print "Scaling weights for cell {}".format(v.name)
      value = sess.run(v.value())
      print value.shape
      print value

      for n in range(0, sample_args.bend_randomly):
        idx = random.randint(0,saved_args.rnn_size-1)
        print "Cell {}".format(idx)
        for j in range(0,4):
          value[:,idx + j * n_cells] *= factor

      sess.run(v.assign(value))


# for v in tf.trainable_variables():
#   print "{} = {}".format(v.name, v.value())
#   value = sess.run(v.value())
#   if v.name == "rnnlm/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix:0" or v.name == "rnnlm/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias:0":
#     modifier = np.zeros(value.shape)
#     sess.run(v.assign(value * modifier))

#   print "Before:"
#   print value

#   modifier = (np.random.random_sample(value.shape) - 0.5) * 0.2 + 0.7

#   sess.run(v.assign(value * modifier))

#   print "After:"
#   print sess.run(v.value())


def generate_sample(prime_points, initial_state):
  prime_data = None
  if sample_args.prime_data is not None:
    prime_start = random.randint(0, len(prime_points)-sample_args.prime_length-1)
    prime_data = prime_points[prime_start:prime_start+sample_args.prime_length]
    
  return model.sample(sess, sample_args.sample_length,sample_args.temperature, prime_data, initial_state, sample_args.distance_filter)

def save_points(points, states, filename):
  with open(filename + ".xyz", "w+") as f:
    last_point = np.array([0,0,0,0,0,0,0])
    for i, p in enumerate(points):
      if sample_args.relative:
        p[0:3] += last_point[0:3]
      
      f.write("{} {} {} {} {} {}\n".format(p[0], p[1], p[2], p[4], p[5], p[6]))

      last_point = p

  if sample_args.writestates:
    with open(filename + ".states", "w+") as statefile:
      for s in states:
        states_as_strings = ["{:.4f}".format(i) for i in s]
        states_as_strings = [(("" if i.startswith("-") else " ") + i) for i in states_as_strings]
        statefile.write(" ".join(states_as_strings) + "\n")

# Generate prime data
prime_points = None
if sample_args.prime_data is not None:
  prime_points = []
  with open(sample_args.prime_data, 'r') as prime_file:
    for line in prime_file:
        point = [float(n) * sample_args.prime_scale for n in line.split(" ") if n.strip() is not ""]
        prime_points.append(point)

if sample_args.concatenate:
  points = None
  states = None
  next_state = None
  for i in range(sample_args.number_sequences):
    print "Generating sequence #{}/{} with {} points.".format(i, sample_args.number_sequences, sample_args.sample_length)
    #new_points, new_states, next_state = generate_sample(prime_points, next_state)
    new_points, new_states, next_state = generate_sample(prime_points, None)
    if points is None:
      points = new_points
      states = new_states
    else:
      points = np.concatenate((points, new_points))
      states = np.concatenate((states, new_states))

  save_points(points, states, sample_args.filename)

else:
  next_state = None
  for i in range(sample_args.number_sequences):
    points, states, next_state = generate_sample(prime_points, next_state)
    save_points(points, states, "{}-{}".format(sample_args.filename, i))


