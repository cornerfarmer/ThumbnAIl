import time

import model.Config
model.Config.db_is_local = True

import tensorflow as tf
from thumbnails import ThumbnailDataset, NormalizedThumbnailDataset

dataset = NormalizedThumbnailDataset(split_in_classes=True)


def image_reader(queue):
    image_reader = tf.WholeFileReader()
    _, image_record = image_reader.read(queue)

    image = tf.image.decode_image(image_record)
    image_crop = tf.reshape(tf.image.crop_to_bounding_box(image, 11, 0, 90 - 11 * 2, 120), (68, 120, 3))

    return tf.cast(image_crop, tf.float32) / 255

with tf.variable_scope("input_pipeline"):
    image_queue = tf.train.string_input_producer(dataset.filenames, shuffle=False)

    image_crop = image_reader(image_queue)

    label_queue = tf.train.input_producer(dataset.labels, shuffle=False)

    batch_size = 64
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    batch_image, batch_label = tf.train.shuffle_batch([image_crop, label_queue.dequeue()], batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)


def network(input, reuse=False):
    with tf.variable_scope("network"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        conv1 = tf.layers.conv2d(input, 32, [5, 5], padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, [2, 2], strides=2)

        conv2 = tf.layers.conv2d(pool1, 64, [5, 5], padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, [2, 2], strides=2)

        pool2_flat = tf.reshape(pool2, [-1, 120 / 4 * 68 / 4 * 64])

        dense = tf.layers.dense(pool2_flat, 1024, tf.sigmoid)
        y = tf.layers.dense(dense, len(dataset.labels[0]))
        if dataset.split_in_classes:
            y_sigm = tf.nn.softmax(y)
        else:
            y_sigm = tf.sigmoid(y)

        return y, y_sigm

y, y_sigm = network(batch_image)

if dataset.split_in_classes:
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=batch_label, logits=y))
else:
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_label, logits=y))
opt = tf.train.AdamOptimizer().minimize(loss)

test_filename = tf.placeholder(tf.string)
test_image_queue = tf.FIFOQueue(capacity=1, dtypes=[tf.string], shapes=[()])
enqueue_op = test_image_queue.enqueue(test_filename)
test_image = tf.expand_dims(image_reader(test_image_queue), 0)
_, test_output = network(test_image, True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

train_writer = tf.summary.FileWriter('tensorboard/' + time.strftime("%Y%m%d-%H%M%S"), sess.graph)

tf.summary.image("original", batch_image)
tf.summary.scalar("loss", loss)
merge_op = tf.summary.merge_all()

print("Max: " + str(dataset.max_views))

for i in xrange(500):
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    _, summary = sess.run([opt, merge_op], run_metadata=run_metadata, options=run_options)
    train_writer.add_run_metadata(run_metadata, 'step%d' % i)

    train_writer.add_summary(summary, i)


def test_video(video_id):
    estimate, _ = sess.run([test_output, enqueue_op], feed_dict={test_filename: "/home/domin/Dokumente/ThumbnAIl/thumbs/" + video_id + ".jpg"})
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

exit()

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))

opt = tf.train.RMSPropOptimizer(0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in xrange(1000):
    _, iter_loss = sess.run((opt, loss), {x: [[1, 0], [0, 0], [0, 1], [1, 1]], y_: [[1], [0], [1], [0]]})
    if i % 100 is 0:
        print(i, iter_loss)

print(sess.run(y_sigm, {x: [[1, 0], [0, 0], [0, 1], [1, 1]]}))


exit()




from model.BaseModel import db, api_key
from model.Video import Video

for video in Video.select():
    print(video.identifier)

print("Finished!")

db.close()