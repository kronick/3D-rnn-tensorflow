import numpy as np
import tensorflow as tf
import random
import argparse
import time
import os
import cPickle

from utils import DataLoader
from model import Model


from RandomWalker import RandomWalker

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--rnn_size', type=int, default=256,
                     help='size of RNN hidden state')
  parser.add_argument('--num_layers', type=int, default=2,
                     help='number of layers in the RNN')
  parser.add_argument('--model', type=str, default='lstm',
                     help='rnn, gru, or lstm')
  parser.add_argument('--batch_size', type=int, default=50,
                     help='minibatch size')
  parser.add_argument('--num_training_samples', type=int, default=10000,
                     help='number of samples to pull from mesh to train on')
  parser.add_argument('--seq_length', type=int, default=300,
                     help='RNN sequence length')
  parser.add_argument('--num_epochs', type=int, default=30,
                     help='number of epochs')
  parser.add_argument('--save_every', type=int, default=500,
                     help='save frequency')
  parser.add_argument('--grad_clip', type=float, default=10.,
                     help='clip gradients at this value')
  parser.add_argument('--learning_rate', type=float, default=0.005,
                     help='learning rate')
  parser.add_argument('--decay_rate', type=float, default=0.95,
                     help='decay rate for rmsprop')
  parser.add_argument('--num_mixture', type=int, default=20,
                     help='number of gaussian mixtures')
  parser.add_argument('--data_scale', type=float, default=100,
                     help='factor to scale raw data up by')
  parser.add_argument('--keep_prob', type=float, default=0.8,
                     help='dropout keep probability')
  parser.add_argument('--mesh_filename', type=str, default="mesh.obj",
                     help='name mesh to use for training data')
  
  relative_parser = parser.add_mutually_exclusive_group(required=False)
  relative_parser.add_argument('--relative', dest='relative', action='store_true')
  relative_parser.add_argument('--absolute', dest='relative', action='store_false')
  parser.set_defaults(relative=True)

  parser.add_argument('--use_checkpoint', type=str, default=None,
                     help='Continue training from a previously saved checkpoint')


  device_parser = parser.add_mutually_exclusive_group(required=False)
  device_parser.add_argument('--gpu', dest='gpu', action='store_true')
  device_parser.add_argument('--cpu', dest='gpu', action='store_false')
  parser.set_defaults(gpu=True)

  args = parser.parse_args()

  device = None if args.gpu is True else '/cpu:0'

  if args.relative:
    print "Using relative coordinates."
  else:
    print "Using absolute coordinates."

  with tf.device(device):
    print("Using device {}".format(device))
    if args.use_checkpoint is not None:
      checkpoint_dir = args.use_checkpoint
      with open(os.path.join(args.use_checkpoint, 'config.pkl')) as f:
        saved_args = cPickle.load(f)
        saved_args.use_checkpoint = checkpoint_dir

      train(saved_args)
    else:
      train(args)

def synthesize_training_data(args):
  # Load mesh walker
  walker = RandomWalker(args.mesh_filename)
  num_batches = args.num_training_samples / args.batch_size
  n_samples = num_batches * args.batch_size
  training_data = walker.walk(n_samples, args.relative, smooth = 0.8)
  training_data[:,0:3] *= args.data_scale

  return training_data

def get_random_batch(data, args):
  x_batch = []
  y_batch = []

  for i in xrange(args.batch_size):
    idx = random.randint(0, len(data)-args.seq_length-2)
    x_batch.append(np.copy(data[idx:idx+args.seq_length]))
    y_batch.append(np.copy(data[idx+1:idx+args.seq_length+1]))

  return (x_batch, y_batch)

def train(args):

    training_data = synthesize_training_data(args)

    with open("train.xyz", "w+") as f:
      last_point = np.array([0,0,0,0])
      for t in training_data:
        p = last_point + t if args.relative else t
        f.write("{} {} {}\n".format(p[0], p[1], p[2]))
        last_point = p
    print "Training data saved ({} points).".format(len(training_data))

    with open(os.path.join('save' if args.use_checkpoint is None else args.use_checkpoint, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    
    model = Model(args)

    num_batches = args.num_training_samples / args.batch_size

    

    with tf.Session() as sess:

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep = 0)
        
        if args.use_checkpoint is not None:
          ckpt = tf.train.get_checkpoint_state(args.use_checkpoint)
          saver.restore(sess, ckpt.model_checkpoint_path)

        for e in xrange(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            state = model.initial_state.eval()
            for b in xrange(num_batches):
                start = time.time()
                x, y = get_random_batch(training_data, args)

                feed = {model.input_data: x, model.target_data: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * num_batches + b,
                            args.num_epochs * num_batches,
                            e, train_loss, end - start)
                if (e * num_batches + b) % args.save_every == 0 and ((e * num_batches + b) > 0):
                    checkpoint_path = os.path.join('save', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * num_batches + b)
                    print "model saved to {}".format(checkpoint_path)

if __name__ == '__main__':
  main()


