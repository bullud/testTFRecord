import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging

num_epochs = 1

batch_size = 1

data_pattern = '/Users/super/yt8m_videofeature/train*.tfrecord'

#files = [gfile.Glob(data_pattern[i]) for i in range(len(data_pattern))]

files = gfile.Glob(data_pattern)
if not files:
    raise IOError("Unable to find training files. data_pattern='" +
                    str(data_pattern) + "'.")

logging.info("Number of training files: %s.", str(len(files)))

filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs, shuffle=True)

reader = tf.TFRecordReader()

_, serialized_examples = reader.read_up_to(filename_queue, batch_size)

feature_map = {"video_id": tf.FixedLenFeature([], tf.string),
               "labels": tf.VarLenFeature(tf.int64),
               "mean_inc3": tf.FixedLenFeature(1024, tf.float32)}


features = tf.parse_example(serialized_examples, features=feature_map)

init_local_op = tf.local_variables_initializer()

with tf.Session() as sess:
    sess.run(init_local_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        #sess.run(features)
        print(serialized_examples.eval())
        #sess.run(serialized_examples)
        #print(len(ex))

    coord.request_stop()
    coord.join()