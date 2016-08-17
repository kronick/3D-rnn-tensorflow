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
parser.add_argument('--scale_factor', type=int, default=10,
                   help='factor to scale down by for svg output.  smaller means bigger output')
parser.add_argument('--temperature', type=float, default=1.0,
                   help='factor to scale standard deviations by, creating bias towards more (>1.0) or less (<1.0) wild output')
parser.add_argument('--prime_index', type=int, default=-1,
                   help='prime sequence generation with a values from a training sample')
parser.add_argument('--prime_sequence_length', type=int, default=1,
                   help='number of stroke sequences use in priming')
parser.add_argument("--number_sequences", type=int, default=1,
                   help='number of writing sequences to generate')
sample_args = parser.parse_args()

with open(os.path.join('save', 'config.pkl')) as f:
    saved_args = cPickle.load(f)

model = Model(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

ckpt = tf.train.get_checkpoint_state('save')
print "loading model: ",ckpt.model_checkpoint_path

# Load training data if we want to prime with a past sequence
prime_array = None
if sample_args.prime_index > -1:
  batch_size = 50
  seq_length = 300
  data_scale = 20
  data_loader = DataLoader(batch_size, seq_length, data_scale)
  prime_array = []
  for i in range(sample_args.prime_sequence_length):
    training_data_sequence = data_loader.get_stroke_sequence_array(sample_args.prime_index + i)
    if training_data_sequence is None:
      print "WARNING: prime_index out of bounds! Not using a priming sequence."
      break

    if len(prime_array) > 0:
      prime_array = np.concatenate([prime_array, training_data_sequence])
    else:
      prime_array = training_data_sequence

  print "Priming with sequence of length {}".format(len(prime_array))


saver.restore(sess, ckpt.model_checkpoint_path)

def sample_stroke(savefile):
  [strokes, params] = model.sample(sess, sample_args.sample_length,sample_args.temperature, prime_array)
  draw_strokes(strokes, factor=sample_args.scale_factor, svg_filename = savefile) 
  #draw_strokes_random_color(strokes, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.color.svg')
  #draw_strokes_random_color(strokes, factor=sample_args.scale_factor, per_stroke_mode = False, svg_filename = sample_args.filename+'.multi_color.svg')
  #draw_strokes_eos_weighted(strokes, params, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.eos_pdf.svg')
  #draw_strokes_pdf(strokes, params, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.pdf.svg')
  return [strokes, params]

for i in range(sample_args.number_sequences):
  [strokes, params] = sample_stroke("{}-{}.svg".format(sample_args.filename, i))


