import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging

data_pattern = ['I:\\testvideo1\\train*.tfrecord', 'I:\\testvideo2\\train*.tfrecord']

num_epochs = 1000

batch_size = 1024

#files = [gfile.Glob(data_pattern[i] for i in range(len(data_pattern)))][0]
files = gfile.Glob(data_pattern)

print(files)

if not files:
    raise IOError("Unable to find training files. data_pattern='" +
                  str(data_pattern[i] for i in range(len(data_pattern))) + "'.")

logging.info("Number of training files: %s.", str(len(files)))
filename_queue = tf.train.string_input_producer(
    files, num_epochs=num_epochs, shuffle=True)

reader = tf.TFRecordReader()

_, serialized_examples = reader.read_up_to(filename_queue, num_records=batch_size)