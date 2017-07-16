import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging

data_pattern = ['/Users/super/yt8m/train*.tfrecord']

num_epochs = 1000

batch_size = 2

num_classes = 4716

files = gfile.Glob(data_pattern)

print(files)

if not files:
    raise IOError("Unable to find training files. data_pattern='" +
                  str(data_pattern[i] for i in range(len(data_pattern))) + "'.")

logging.info("Number of training files: %s.", str(len(files)))

filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs, shuffle=True)

reader = tf.TFRecordReader()

_, serialized_examples = reader.read_up_to(filename_queue, num_records=batch_size)

feature_map = {"video_id": tf.FixedLenFeature([], tf.string),
               "labels": tf.VarLenFeature(tf.int64),
               "mean_rgb": tf.FixedLenFeature(1024, tf.float32)}

features = tf.parse_example(serialized_examples, features=feature_map)



init_local_op = tf.initialize_local_variables()

with tf.Session() as sess:
    sess.run(init_local_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    total_step = 0
    while total_step < 10:
        examples = sess.run(serialized_examples)
        print(examples.shape)

        #feas = sess.run(features)
        #print(feas['video_id'])
        #print(feas['labels'])

        #features = tf.parse_example(examples, features=feature_map)

        feas = sess.run(features)

        video_id = feas['video_id']
        print('video_id = ' + video_id)

        labels_num = len(feas['labels'])
        print('labels_num = ' + str(labels_num))

        labels = tf.sparse_to_indicator(feas["labels"], num_classes)

        labels.set_shape([None, num_classes])

        labs = sess.run(labels)

        print(labs)

        total_step = total_step + 1

    coord.request_stop()
    coord.join(threads)