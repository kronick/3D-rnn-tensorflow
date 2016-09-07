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
# for v in tf.all_variables():
#   #print "{} = {}".format(v.name, v.value())
#   if v.name.startswith("rnnlm/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix:0"):
#     print "{}: {}".format(v.name, v.value())
#     for n in sess.run(v.value())[0]:
#       print n
  


def generate_sample(prime_points, initial_state):
  prime_data = None
  if sample_args.prime_data is not None:
    prime_start = random.randint(0, len(prime_points)-sample_args.prime_length-1)
    prime_data = prime_points[prime_start:prime_start+sample_args.prime_length]
  return model.sample(sess, sample_args.sample_length,sample_args.temperature, prime_data, initial_state, sample_args.distance_filter)

def save_points(points, filename):
  with open(filename + ".xyz", "w+") as f:
    with open(filename + ".json", "w+") as jsonfile:
      jsonfile.write("pathPoints = [\n")

      last_point = np.array([0,0,0,0,0,0,0])
      for i, p in enumerate(points):
        if sample_args.relative:
          p[0:3] += last_point[0:3]
        
        f.write("{} {} {} {} {} {}\n".format(p[0], p[1], p[2], p[4], p[5], p[6]))
        jsonfile.write("\t[{}, {}, {}, {}, {}, {}]{}\n".format(p[0], p[1], p[2], p[4], p[5], p[6], ',' if i < len(points) else ''))

        last_point = p

      jsonfile.write("];")  


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
  next_state = None
  for i in range(sample_args.number_sequences):
    print "Generating sequence #{}/{} with {} points.".format(i, sample_args.number_sequences, sample_args.sample_length)
    new_points, next_state = generate_sample(prime_points, next_state)
    if points is None:
      points = new_points
    else:
      points = np.concatenate((points, new_points))

  save_points(points, sample_args.filename)

else:
  next_state = None
  for i in range(sample_args.number_sequences):
    points, next_state = generate_sample(prime_points, next_state)
    save_points(points, "{}-{}".format(sample_args.filename, i))


