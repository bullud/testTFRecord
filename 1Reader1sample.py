import tensorflow as tf

filenames = ['A.csv', 'B.csv', 'C.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle = False)

reader = tf.TextLineReader()
#key, value = reader.read(filename_queue)
key, value = reader.read_up_to(filename_queue, 2)

example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        print(example.eval())

    coord.request_stop()
    coord.join(threads)

