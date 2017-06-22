import tensorflow as tf
import numpy as np

a = tf.SparseTensor(indices = [[0, 1], [0, 3], [2, 0]], values=[1,2,3], dense_shape=[3, 5])
b = tf.sparse_to_indicator(a, 10)
sess = tf.Session()
print(sess.run(b))
sess.close()