"""
@author: Arjun Krishna
@desc: data-manager
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer

class DataManager:

  def __init__(self, data_path):
    self.data_path  = data_path

    with open(os.path.join(data_path, 'train.txt')) as f:
      train_data = f.read().lower().replace('\n', '<eos>').split()

    self.vocab = train_data
    self.vocab_size = len(self.vocab)

    self.id2word = dict(zip(range(self.vocab_size), self.vocab))
    self.word2id = dict(zip(self.vocab, range(self.vocab_size)))

    self.raw_train_data = [self.word2id[w] for w in train_data if w in self.word2id]

  def get_train(self, batch_size, num_steps):

    with tf.name_scope("Data_Producer", [self.raw_train_data, batch_size, num_steps]):
      self.train_data = tf.convert_to_tensor(self.raw_train_data, name="train_data", dtype=tf.int32)

      data_len = tf.size(self.train_data)
      batch_len = data_len // batch_size

      data = tf.reshape(self.train_data[0 : batch_size*batch_len],
                        [batch_size, batch_len])
      epoch_size = (batch_len - 1) // num_steps

      self.epoch_size = epoch_size

      assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")

      with tf.control_dependencies([assertion]):
        epoch_size = tf.identity(epoch_size, name="epoch_size")

      i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

      x = tf.strided_slice(data, [0, i * num_steps],
                           [batch_size, (i + 1) * num_steps])
      x.set_shape([batch_size, num_steps])
      y = tf.strided_slice(data, [0, i * num_steps + 1],
                           [batch_size, (i + 1) * num_steps + 1])
      y.set_shape([batch_size, num_steps])

      return x, y