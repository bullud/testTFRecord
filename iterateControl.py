import tensorflow as tf

filenames = ['A.csv', 'B.csv', 'C.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=3)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [['null'], ['null']]
example_list = [tf.decode_csv(value, record_defaults=record_defaults) for _ in range(2)]

example_batch, label_batch = tf.train.batch_join(example_list, batch_size=5)

init_local_op = tf.initialize_local_variables()

with tf.Session() as sess:
    sess.run(init_local_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop():
            print(example_batch.eval())
    except tf.errors.OutOfRangeError:
        print('Epochs Complete!')
    finally:
        coord.request_stop()

    #coord.join(threads)
    coord.request_stop()
    coord.join(threads)