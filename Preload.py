import tensorflow as tf

x1 = tf.constant([2, 3, 4])
x2 = tf.constant([4, 0, 1])
y = tf.add(x1, x2)

with tf.Session() as sess:
    print(sess.run(y))



