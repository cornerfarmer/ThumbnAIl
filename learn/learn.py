import time

import model.Config
model.Config.db_is_local = True

import tensorflow as tf
from thumbnails import ThumbnailDataset, NormalizedThumbnailDataset

dataset = NormalizedThumbnailDataset(split_in_classes=True, max_per_set=None)


def image_reader(queue):
    image_reader = tf.WholeFileReader()
    _, image_record = image_reader.read(queue)

    image = tf.image.decode_image(image_record)
    image_crop = tf.reshape(tf.image.crop_to_bounding_box(image, 11, 0, 90 - 11 * 2, 120), (68, 120, 3))

    return tf.cast(image_crop, tf.float32) / 255


def data_augmentation(image):
    with tf.variable_scope("data_augmentation"):
        width = 120
        height = 68
        image = tf.image.random_flip_left_right(image)

        dx = tf.random_uniform([], -20, 20, dtype=tf.int32)
        dy = tf.random_uniform([], -20, 20, dtype=tf.int32)
        image = tf.image.crop_to_bounding_box(image, tf.maximum(0, -dy), tf.maximum(0, -dx), height - tf.abs(dy), width - tf.abs(dx))
        image = tf.image.pad_to_bounding_box(image, tf.maximum(0, dy), tf.maximum(0, dx), height, width)

        image = tf.contrib.image.rotate(image, tf.random_uniform([1], -0.5, 0.5))

    return image


def input_pipeline(filenames, labels, enable_augmentation=False):
    image_queue = tf.train.string_input_producer(filenames, shuffle=False)

    image_crop = image_reader(image_queue)

    if enable_augmentation:
        image_crop = data_augmentation(image_crop)

    label_queue = tf.train.input_producer(labels, shuffle=False)

    batch_size = 64
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    return tf.train.shuffle_batch([image_crop, label_queue.dequeue()], batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)


with tf.variable_scope("input_train_pipeline"):
    train_batch_image, train_batch_label = input_pipeline(dataset.train_filenames, dataset.train_labels, True)

with tf.variable_scope("input_test_pipeline"):
    test_batch_image, test_batch_label = input_pipeline(dataset.test_filenames, dataset.test_labels)

def network(input, keep_prob, reuse=False):
    with tf.variable_scope("network"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        conv1 = tf.layers.conv2d(input, 32, [3, 3], padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, [2, 2], strides=2)
       # pool1 = tf.layers.batch_normalization(pool1, axis=1, training=not reuse)

        conv2 = tf.layers.conv2d(pool1, 64, [3, 3], padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, [2, 2], strides=2)
       # pool2 = tf.layers.batch_normalization(pool2, axis=1, training=not reuse)

        conv3 = tf.layers.conv2d(pool2, 128, [3, 3], padding='same', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3, [2, 2], strides=2)
       # pool3 = tf.layers.batch_normalization(pool3, axis=1, training=not reuse)

        flat = tf.reshape(pool3, [-1, 120 / 8 * int(68 / 8) * 128])

        dense = tf.layers.dense(flat, 200, tf.nn.relu)
       # dense = tf.layers.batch_normalization(dense, training=not reuse)
        dense_drop = tf.nn.dropout(dense, keep_prob)

        y = tf.layers.dense(dense_drop, len(dataset.labels[0]))
        if dataset.split_in_classes:
            y_act = tf.nn.softmax(y)
        else:
            y_act = tf.sigmoid(y)

        return y, y_act

train_y, train_y_act = network(train_batch_image, 0.5)
test_y, test_y_act = network(test_batch_image, 1, reuse=True)

with tf.variable_scope("loss"):
    if dataset.split_in_classes:
        train_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=train_batch_label, logits=train_y))
        test_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=test_batch_label, logits=test_y))
    else:
        train_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_batch_label, logits=train_y))
        test_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=test_batch_label, logits=test_y))

with tf.variable_scope("accuracy"):
    train_correct_prediction = tf.equal(tf.argmax(train_batch_label, 1), tf.argmax(train_y, 1))
    train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
    test_correct_prediction = tf.equal(tf.argmax(test_batch_label, 1), tf.argmax(test_y, 1))
    test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))

with tf.variable_scope("optimizer"):
    opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(train_loss)

with tf.variable_scope("validation"):
    validation_filename = tf.placeholder(tf.string)
    validation_image_queue = tf.FIFOQueue(capacity=1, dtypes=[tf.string], shapes=[()])
    enqueue_op = validation_image_queue.enqueue(validation_filename)
    validation_image = tf.expand_dims(image_reader(validation_image_queue), 0)

_, validation_output = network(validation_image, 1, True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

train_writer = tf.summary.FileWriter('tensorboard/' + time.strftime("%Y%m%d-%H%M%S"), sess.graph)

tf.summary.image("original", train_batch_image)
with tf.variable_scope("eval"):
    tf.summary.scalar("train_loss", train_loss)
    tf.summary.scalar("test_loss", test_loss)
    tf.summary.scalar("train_acc", train_accuracy)
    tf.summary.scalar("test_acc", test_accuracy)
merge_op = tf.summary.merge_all()

print("Train: " + str(len(dataset.train_filenames)))
print("Test: " + str(len(dataset.test_filenames)))

for i in xrange(1500):
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    _, summary = sess.run([opt, merge_op], run_metadata=run_metadata, options=run_options)

    train_writer.add_run_metadata(run_metadata, 'step%d' % i)

    train_writer.add_summary(summary, i)


def test_video(video_id):
    estimate, _ = sess.run([validation_output, enqueue_op], feed_dict={validation_filename: "/home/domin/Dokumente/ThumbnAIl/thumbs/" + video_id + ".jpg"})
    estimate = estimate[0]
    print(estimate)
    estimate = dataset.calculate_views_from_label(estimate if dataset.split_in_classes else estimate[0])
    real = dataset.get_view_count_for_video(video_id)
    print("Video " + video_id + ":  est: " + str(estimate) + "  real: " + str(real) + "  diff: " + str(estimate - real))


test_video("_1PMlT8vmiA")
test_video("4gSOMba1UdM")
test_video("EeZsEh1GB2Q")

for video_set in dataset.videos:
    test_video(video_set[0].identifier)
