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
parser.add_argument('--scale_factor', type=int, default=5,
                   help='factor to scale down by for svg output.  smaller means bigger output')
parser.add_argument('--temperature', type=float, default=1.0,
                   help='factor to scale standard deviations by, creating bias towards more (>1.0) or less (<1.0) wild output')
parser.add_argument("--number_sequences", type=int, default=1,
                   help='number of writing sequences to generate')

relative_parser = parser.add_mutually_exclusive_group(required=False)
relative_parser.add_argument('--relative', dest='relative', action='store_true')
relative_parser.add_argument('--absolute', dest='relative', action='store_false')
parser.set_defaults(relative=False)

sample_args = parser.parse_args()

with open(os.path.join('save', 'config.pkl')) as f:
    saved_args = cPickle.load(f)

model = Model(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

ckpt = tf.train.get_checkpoint_state('save')
print "loading model: ",ckpt.model_checkpoint_path

saver.restore(sess, ckpt.model_checkpoint_path)

def sample_stroke(savefile):
  points = model.sample(sess, sample_args.sample_length,sample_args.temperature)

  with open(savefile, "w+") as f:
    with open(savefile + ".json", "w+") as jsonfile:
      jsonfile.write("pathPoints = [\n")

      last_point = np.array([0,0,0,0,0,0,0])
      for i, p in enumerate(points):
        if sample_args.relative:
          p[0:3] += last_point[0:3]
        
        f.write("{} {} {} {} {} {}\n".format(p[0], p[1], p[2], p[4], p[5], p[6]))
        jsonfile.write("\t[{}, {}, {}, {}, {}, {}]{}\n".format(p[0], p[1], p[2], p[4], p[5], p[6], ',' if i < len(points) else ''))

        last_point = p

      jsonfile.write("];")

  return points

for i in range(sample_args.number_sequences):
  strokes = sample_stroke("{}-{}.xyz".format(sample_args.filename, i))


