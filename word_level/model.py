"""
@author: Arjun Krishna
@desc: A character-level Language Model using LSTM
"""
import tensorflow as tf

class Model:

  def __init__(self, input_data, targets, config, is_training=False):
    self._is_training = is_training
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps

    size = config.hidden_size
    vocab_size = config.vocab_size

    embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)

    inputs = tf.nn.embedding_lookup(embedding, input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    
    with tf.variable_scope('multi_rnn_cell'):

      def make_cell():
        cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, 
                                            state_is_tuple=True, reuse=not is_training)
        if is_training and config.keep_prob < 1:
          cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
        return cell

      cell = tf.contrib.rnn.MultiRNNCell(
                [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

      self._initial_state = cell.zero_state(config.batch_size, tf.float32)

      state = self._initial_state

      inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
      outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=self._initial_state)

      output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])

    with tf.variable_scope('fc1'):
      W = tf.get_variable(
            "W", shape=[size, vocab_size], initializer=tf.contrib.layers.xavier_initializer())
      b = tf.get_variable(
            "b", shape=[vocab_size], initializer=tf.zeros_initializer())

      logits = tf.matmul(output, W) + b

      logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

      self._prediction = tf.nn.softmax(logits, name="prediction")
    
    with tf.name_scope('loss'):
      self._loss = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
                      logits,
                      targets,
                      tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
                      average_across_timesteps=False,
                      average_across_batch=True))

      self._final_state = state
    
    if not is_training:
      return

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(config.lr)

    self._optimize = optimizer.apply_gradients(
                        zip(grads, tvars), 
                        global_step=tf.train.get_or_create_global_step()
                     )
    
  @property
  def initial_state(self):
    return self._initial_state

  @property
  def final_state(self):
    return self._final_state

  @property
  def prediction(self):
    return self._prediction

  @property
  def optimize(self):
    return self._optimize

  @property
  def loss(self):
    return self._loss
