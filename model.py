import tensorflow as tf

import numpy as np
import random

class Model():
  def __init__(self, args, infer=False):
    self.args = args

    COORDINATE_DIMENSIONS = 3

    if infer:
      args.batch_size = 1
      args.seq_length = 1

    if args.model == 'rnn':
      cell_fn = tf.nn.rnn_cell.BasicRNNCell
    elif args.model == 'gru':
      cell_fn = tf.nn.rnn_cell.GRUCell
    elif args.model == 'lstm':
      cell_fn = tf.nn.rnn_cell.BasicLSTMCell
    else:
      raise Exception("model type not supported: {}".format(args.model))

    cell = cell_fn(args.rnn_size)

    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)

    if (infer == False and args.keep_prob < 1): # training mode
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = args.keep_prob)

    self.cell = cell

    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, COORDINATE_DIMENSIONS + 1])
    self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, COORDINATE_DIMENSIONS + 1])
    self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

    self.num_mixture = args.num_mixture
    #NOUT = 1 + self.num_mixture * 6 # end_of_stroke + prob + 2*(mu + sig) + corr
    NOUT = 1 + self.num_mixture * (2 + COORDINATE_DIMENSIONS) # end_of_stroke + mixtures * (weight + std deviation + COORDINATE_DIMENSIONS*mean)

    with tf.variable_scope('rnnlm'):
      output_w = tf.get_variable("output_w", [args.rnn_size, NOUT])
      output_b = tf.get_variable("output_b", [NOUT])

    inputs = tf.split(1, args.seq_length, self.input_data)
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope='rnnlm')
    output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    self.final_state = last_state

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape(self.target_data,[-1, COORDINATE_DIMENSIONS + 1])

    [x1_data, x2_data, x3_data, eos_data] = tf.split(1, COORDINATE_DIMENSIONS + 1, flat_target_data)

    # long method:
    #flat_target_data = tf.split(1, args.seq_length, self.target_data)
    #flat_target_data = [tf.squeeze(flat_target_data_, [1]) for flat_target_data_ in flat_target_data]
    #flat_target_data = tf.reshape(tf.concat(1, flat_target_data), [-1, 3])

    def tf_3d_normal(x1, x2, x3, mu1, mu2, mu3, sigma):
      # Probability distribution function for iosotropic multivariate gaussian at point (x1, x2, x3)
      # Isotropic because we have only 1 variance/sigma for all dimensions.
      # eq #23 of Bishop's Mixture Density Networks paper. He explains why this isotropic simplification
      # is OK because we are using a _mixture_ of gaussians as long as we use enough mixtures
      # TODO: Try making this adaptable to non-isotropic model (require fewer mixtures?)
      norm1 = tf.square(tf.sub(x1, mu1))
      norm2 = tf.square(tf.sub(x2, mu2))
      norm3 = tf.square(tf.sub(x3, mu3))
      
      Z = norm1 + norm2 + norm3
      numerator = tf.exp(tf.div(-Z, 2 * tf.square(sigma)))
      denominator = tf.pow(2 * np.pi, 3./2.) * tf.pow(sigma, 3)

      resul = tf.div(numberator, denominator)
      
      return result

    def get_lossfunc(z_pi, z_mu1, z_mu2, z_mu3, z_sigma, z_eos, x1_data, x2_data, x3_data, eos_data):
      result0 = tf_3d_normal(x1_data, x2_data, x3_data, z_mu1, z_mu2, z_mu3, z_sigma)
      # implementing eq # 26 of http://arxiv.org/abs/1308.0850
      epsilon = 1e-20
      result1 = tf.mul(result0, z_pi)
      result1 = tf.reduce_sum(result1, 1, keep_dims=True)
      result1 = -tf.log(tf.maximum(result1, 1e-20)) # at the beginning, some errors are exactly zero.

      result2 = tf.mul(z_eos, eos_data) + tf.mul(1-z_eos, 1-eos_data) # handles the IF condition in the paper
      result2 = -tf.log(result2)

      result = result1 + result2
      return tf.reduce_sum(result)

    # below is where we need to do MDN splitting of distribution params
    def get_mixture_coef(output):
      # returns the tf slices containing mdn dist params
      # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
      z = output
      z_eos = z[:, 0:1]
      z_pi, z_mu1, z_mu2, z_mu3, z_sigma = tf.split(1, 5, z[:, 1:])

      # process output z's into MDN paramters

      # end of stroke signal
      z_eos = tf.sigmoid(z_eos) # should be negated, but doesn't matter.

      # softmax all the pi's:
      max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
      z_pi = tf.sub(z_pi, max_pi)
      z_pi = tf.exp(z_pi)
      normalize_pi = tf.inv(tf.reduce_sum(z_pi, 1, keep_dims=True))
      z_pi = tf.mul(normalize_pi, z_pi)

      # exponentiate the sigma to make sure it's positive
      z_sigma = tf.exp(z_sigma)

      return [z_pi, z_mu1, z_mu2, z_mu3, z_sigma, z_eos]

    [o_pi, o_mu1, o_mu2, o_mu3, o_sigma, o_eos] = get_mixture_coef(output)

    self.pi = o_pi
    self.mu1 = o_mu1
    self.mu2 = o_mu2
    self.mu3 = o_mu3
    self.sigma = o_sigma
    self.eos = o_eos

    lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_mu3, o_sigma, o_eos, x1_data, x2_data, x3_data, eos_data)
    self.cost = lossfunc / (args.batch_size * args.seq_length)

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))


  def sample(self, sess, num=1200, scale_sigma = 1.0, prime_array = None):

    def get_pi_idx(x, pdf):
      """ Used to choose a random component from the weighted set of gaussian mixtures
          Returns the index of the chosen mixture """

      N = pdf.size
      accumulate = 0
      for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
          return i
      print 'error with sampling ensemble'
      return -1

    def sample_gaussian_3d(mu1, mu2, mu3, s1, s2, s3):
      # Draw a sample from 3D Gaussian distribution. Only supports diagonal covariance matrix for now
      mean = [mu1, mu2, mu3]
      cov = [[s1, 0, 0], [0, s2, 0], [0, 0, s3]]
      x = np.random.multivariate_normal(mean, cov, 1)
      return x[0][0], x[0][1], x[0][2]

    # Set up starting conditions 
    prev_x = np.zeros((1, 1, COORDINATE_DIMENSIONS + 1), dtype=np.float32)
    prev_x[0, 0, COORDINATE_DIMENSIONS] = 1 # initially, we want to see beginning of new stroke
    prev_state = sess.run(self.cell.zero_state(1, tf.float32))


    prime_stroke_count = 0 if prime_array is None else len(prime_array)

    strokes = np.zeros((num + prime_stroke_count, COORDINATE_DIMENSIONS + 1), dtype=np.float32)
    mixture_params = []

    # Feed in the priming array if provided 
    if prime_array is not None:
      i = 0
      for point in prime_array:
        
        feed = {self.input_data: prev_x, self.initial_state: prev_state}
        # Get new output based on previous step
        [o_pi, o_mu1, o_mu2, o_mu3, o_sigma, o_eos, next_state] = sess.run([self.pi, self.mu1, self.mu2, self.mu3, self.sigma, self.eos, self.final_state],feed)
        # Skip the parts where we calculate next point from MDN because we should just use the next point in the prime training sequence

        # But propogate the state forward
        prev_x = point.reshape(1,1,3)
        prev_state = next_state

        # And add stroke to output
        strokes[i,:] = point
        params = [o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0], o_eos[0]]
        mixture_params.append(params)
        i += 1

    for i in xrange(num):
      # Each time step

      feed = {self.input_data: prev_x, self.initial_state:prev_state}

      # Get new output based on previous step
      [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, next_state] = sess.run([self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.eos, self.final_state],feed)

      # Choose a guassian distribution to sample from based on their weights
      idx = get_pi_idx(random.random(), o_pi[0])

      # Chance of picking up pen at each step
      eos = 1 if random.random() < o_eos[0][0] else 0

      # Calculate a weighted random next point according to the chosen gaussian distribution
      # Scale the standard deviations to bias towards more or less "normal" output
      sig1 = sig2 = sig3 = scale_sigma * o_sigma[0][idx]
      next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx], o_mus[0][idx], sig1, sig2, sig3)

      strokes[i + prime_stroke_count,:] = [next_x1, next_x2, eos]

      params = [o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0], o_eos[0]]
      mixture_params.append(params)

      prev_x = np.zeros((1, 1, 3), dtype=np.float32)
      prev_x[0][0] = np.array([next_x1, next_x2, eos], dtype=np.float32)
      prev_state = next_state

    strokes[:,0:2] *= self.args.data_scale
    return strokes, mixture_params
