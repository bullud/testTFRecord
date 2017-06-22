import tensorflow as tf

x1 = tf.placeholder(tf.int16)
x2 = tf.placeholder(tf.int16)
y = tf.add(x1, x2)

li1 = [2, 3, 4]
li2 = [4, 0, 1]

with tf.Session() as sess:
    print(sess.run(y, feed_dict={x1: li1, x2: li2}))

