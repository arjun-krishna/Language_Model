"""
@author: Arjun Krishna
@desc: Language Model (word-level)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import os
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'Mode of application : train (or) sample')
flags.DEFINE_integer('num_epochs', 15, 'Numer of epochs')
flags.DEFINE_string('log_path', 'log_dir', 'Directory to log summaries')
flags.DEFINE_string('save_path', 'chkpt', 'Directory to save/restore model')
flags.DEFINE_string('data_path', '../data/', 'Directory from which the data is read')

flags.DEFINE_integer('batch_size', 64, 'Batch Size')
flags.DEFINE_integer('num_steps', 20, 'Num time steps')
flags.DEFINE_integer('gen_len', 400, 'generate text of gen_len characters')
flags.DEFINE_boolean('verbose', False, 'verbose')

FLAGS = flags.FLAGS

from model import Model
from data import DataManager

class Config:

  batch_size = None
  num_steps = None
  vocab_size = None
  keep_prob = 0.9
  hidden_size = 200
  num_layers = 2
  max_grad_norm = 5
  lr = 1


def main(_):

  dm = DataManager(FLAGS.data_path)

  train_config = Config()
  train_config.vocab_size = dm.vocab_size
  train_config.batch_size = FLAGS.batch_size
  train_config.num_steps = FLAGS.num_steps

  with tf.name_scope("Train"):
    train_inputs, train_targets = dm.get_train(FLAGS.batch_size, FLAGS.num_steps)

    with tf.variable_scope("Model", reuse=None):
      model = Model(train_inputs, train_targets, train_config, is_training=True)
    tf.summary.scalar('loss', model.loss)

  sampler_config = Config()
  sampler_config.vocab_size = dm.vocab_size
  sampler_config.batch_size = 1
  sampler_config.num_steps = 1

  with tf.name_scope("Sampler"):
    inputs = tf.placeholder(tf.int32, shape=[1, 1], name="sampler_inputs")
    target = tf.placeholder(tf.int32, shape=[1, 1], name="sampler_target")

    with tf.variable_scope("Model", reuse=True):
      sampler = Model(inputs, target, sampler_config, is_training=False)

  merged = tf.summary.merge_all()

  saver = tf.train.Saver()
  
  FLAGS.mode = 1 if FLAGS.mode == 'train' else 0

  with tf.Session() as sess:

    if FLAGS.mode:
      summary_writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)
      tf.global_variables_initializer().run()
      
      epoch_size = (len(dm.raw_train_data) // FLAGS.batch_size - 1) // FLAGS.num_steps

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)


      pbar = tqdm(total=FLAGS.num_epochs)
      for epoch in range(FLAGS.num_epochs):
        
        state = sess.run(model.initial_state)

        total_loss = 0.0
        iters = 0

        for step in range(epoch_size):

          feed_dict = {}
          for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

          loss, state, summary, _ = sess.run([model.loss, model.final_state, merged, model.optimize], feed_dict)

          summary_writer.add_summary(summary, step + epoch_size*epoch)

          total_loss += loss
          iters += FLAGS.num_steps

          if FLAGS.verbose and step % (epoch_size // 10) == 10:
            print ("%.3f perplexity: %.3f" % (step* 1.0/ epoch_size, np.exp(total_loss/ iters))) 

        if FLAGS.verbose:
          print ("epoch %d | perplexity: %.3f" % (epoch+1, np.exp(total_loss/ iters)))

        save_path = saver.save(sess, os.path.join(FLAGS.save_path, "model.ckpt"))
        pbar.update(1)
      pbar.close()

    else:
    	saver.restore(sess, os.path.join(FLAGS.save_path, "model.ckpt"))

    # simply test (works with train and sample mode)
    # =========================================================================================

    def generate_text():
      state = sess.run(sampler.initial_state)
      
      word = np.random.choice(dm.vocab_size, (1,1))

      generated_txt = ''

      for i in range(FLAGS.gen_len):
        feed_dict = {}
        for i, (c,h) in enumerate(sampler.initial_state):
          feed_dict[c] = state[i].c
          feed_dict[h] = state[i].h

        feed_dict[inputs] = word

        pred, state = sess.run([sampler.prediction, sampler.final_state], feed_dict=feed_dict)

        word_id = np.argmax(pred[0][0])

        generated_txt += (dm.id2word[word_id] + ' ')

        word = np.array([[word_id]])

      return generated_txt

    print (generate_text())

    # =========================================================================================

    if FLAGS.mode:
    	save_path = saver.save(sess, os.path.join(FLAGS.save_path, "model.ckpt"))
    	print ('Model saved in %s' % save_path)

if __name__ == "__main__":
  tf.app.run()
