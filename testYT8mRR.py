import json
import os
import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
from tensorflow import flags
import tensorflow.contrib.slim as slim
import losses
import utils
import time
import eval_util

#data_pattern = ['/Users/super/yt8m/train*.tfrecord']

#data_pattern = ['/Users/super/yt8m_videofeature/train*.tfrecord']

data_pattern = ['I:\\yt8m_video\\train*.tfrecord']

num_epochs = 5

reader_batch_size = 1024

num_readers = 1

num_classes = 4716

mini_batch_size = 1024

label_loss_fn=losses.CrossEntropyLoss()

regularization_penalty = 1

base_learning_rate = 0.01

learning_rate_decay_examples = 4000000

learning_rate_decay = 0.95

optimizer_class = tf.train.AdamOptimizer

clip_gradient_norm = 1.0

regularization_penalty = 1

num_towers = 1

max_steps = 100000

def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)

def get_reader(filename_queue, reader_batch_size, num_classes):

    reader = tf.TFRecordReader()

    _, serialized_examples = reader.read_up_to(filename_queue, num_records=reader_batch_size)

    feature_map = {"video_id": tf.FixedLenFeature([], tf.string),
                   "labels": tf.VarLenFeature(tf.int64),
                   "mean_rgb": tf.FixedLenFeature(1024, tf.float32)}

    features = tf.parse_example(serialized_examples, features=feature_map)

    video_ids = features['video_id']
    labels = tf.sparse_to_indicator(features["labels"], num_classes)
    labels.set_shape([None, num_classes])

    feature_dim = len(features['mean_rgb'].get_shape()) - 1

    video_features = tf.nn.l2_normalize(features['mean_rgb'], dim=feature_dim)

    return video_ids, video_features, labels, tf.ones([tf.shape(video_ids)[0]])

def gen_input(data_pattern, reader_batch_size, num_classes, num_readers, mini_batch_size):

    files = gfile.Glob(data_pattern)
    files.sort()

    if not files:
        raise IOError("Unable to find training files. data_pattern='" +
                      str(data_pattern[i] for i in range(len(data_pattern))) + "'.")

    logging.info("Number of training files: %s.", str(len(files)))

    filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs, shuffle=True)
    #return filename_queue.dequeue()

    training_data = [get_reader(filename_queue, reader_batch_size, num_classes) for _ in range(num_readers)]

    return tf.train.shuffle_batch_join(training_data, batch_size = mini_batch_size,
                                capacity = mini_batch_size*5, min_after_dequeue=mini_batch_size,
                                allow_smaller_final_batch=True, enqueue_many=True)

    #return tf.train.batch_join(training_data, batch_size=mini_batch_size,
    #                            capacity = mini_batch_size*5,
    #                            allow_smaller_final_batch=True, enqueue_many=True)

def gen_model(model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))

    return {"predictions": output}


def main():
    env = json.loads(os.environ.get("TF_CONFIG", "{}"))

    task_data = env.get("task", None) or {"type": "master", "index": 0}
    task = type("TaskSpec", (object,), task_data)

    logging.set_verbosity(tf.logging.INFO)
    logging.info("%s: Tensorflow version: %s.",
                 task_as_string(task), tf.__version__)

    video_ids, video_features, video_labels, video_frames = gen_input(data_pattern, reader_batch_size=reader_batch_size,
                            num_classes=num_classes, num_readers=num_readers, mini_batch_size=mini_batch_size)

    result = gen_model(model_input=video_features, vocab_size=num_classes, labels=video_labels, num_frames=video_frames)

    predictions = result["predictions"]

    global_step = tf.Variable(0, trainable=False, name="global_step")

    label_loss = label_loss_fn.calculate_loss(predictions, video_labels)

    if "regularization_loss" in result.keys():
        reg_loss = result["regularization_loss"]
    else:
        reg_loss = tf.constant(0.0)

    reg_losses = tf.losses.get_regularization_losses()
    if reg_losses:
        reg_loss += tf.add_n(reg_losses)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if "update_ops" in result.keys():
        update_ops += result["update_ops"]

    if update_ops:
        with tf.control_dependencies(update_ops):
            barrier = tf.no_op(name="gradient_barrier")
            with tf.control_dependencies([barrier]):
                label_loss = tf.identity(label_loss)

    final_loss = regularization_penalty * reg_loss + label_loss

    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step * mini_batch_size * num_towers,
        learning_rate_decay_examples,
        learning_rate_decay,
        staircase=True)

    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = optimizer_class(learning_rate)

    gradients = optimizer.compute_gradients(final_loss,
                                            colocate_gradients_with_ops=False)

    tf.summary.scalar("label_loss", label_loss)

    tf.summary.scalar("reg_loss", reg_loss)

    if clip_gradient_norm > 0:
        with tf.name_scope('clip_grads'):
            gradients = utils.clip_gradient_norms(gradients, clip_gradient_norm)

    train_op = optimizer.apply_gradients(gradients, global_step=global_step)


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        #init_local_op = tf.local_variables_initializer()
        #sess.run(init_local_op)

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(coord=coord)

        total_step = 0

        try:
            while total_step < 100000:
                batch_start_time = time.time()

                # v_ids, v_features, v_labels, v_frames = sess.run([video_ids, video_features, video_labels, video_frames])

                _, global_step_val, loss_val, predictions_val, labels_val = sess.run(
                    [train_op, global_step, label_loss, predictions, tf.cast(video_labels, tf.float32)])

                seconds_per_batch = time.time() - batch_start_time
                examples_per_second = labels_val.shape[0] / seconds_per_batch

                # if max_steps <= global_step_val:
                #    max_steps_reached = True
                # print(v_features.shape)
                # print(v_ids)

                if total_step % 10 == 0:
                    eval_start_time = time.time()
                    hit_at_one = eval_util.calculate_hit_at_one(predictions_val, labels_val)
                    perr = eval_util.calculate_precision_at_equal_recall_rate(predictions_val,
                                                                              labels_val)
                    gap = eval_util.calculate_gap(predictions_val, labels_val)
                    eval_end_time = time.time()
                    eval_time = eval_end_time - eval_start_time

                    logging.info("training step " + str(global_step_val) + " | Loss: " + ("%.2f" % loss_val) +
                                 " Examples/sec: " + ("%.2f" % examples_per_second) + " | Hit@1: " +
                                 ("%.2f" % hit_at_one) + " PERR: " + ("%.2f" % perr) +
                                 " GAP: " + ("%.2f" % gap))

                else:
                    logging.info("training step " + str(global_step_val) + " | Loss: " +
                                 ("%.2f" % loss_val) + " Examples/sec: " + ("%.2f" % examples_per_second))

                total_step = total_step + 1

        except tf.errors.OutOfRangeError:
            logging.info("%s: Done training -- epoch limit reached.", task_as_string(task))

        coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    main()
