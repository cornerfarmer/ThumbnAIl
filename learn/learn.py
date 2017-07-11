
import model.Config
model.Config.db_is_local = True

import tensorflow as tf
from thumbnails import ThumbnailDataset

dataset = ThumbnailDataset()

image_queue = tf.train.string_input_producer(dataset.filenames, shuffle=False)

image_reader = tf.WholeFileReader()
_, image_record = image_reader.read(image_queue)

image = tf.image.decode_jpeg(image_record)
image_crop = tf.reshape(tf.image.crop_to_bounding_box(image, 11, 0, 90 - 11 * 2, 120), (68, 120, 3))

label_queue = tf.train.input_producer(dataset.labels, shuffle=False)

batch_size = 32
min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
batch_image, batch_label = tf.train.shuffle_batch([image_crop, label_queue.dequeue()], batch_size, capacity, min_after_dequeue)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

image = sess.run([batch_image, batch_label])

print(image)

exit()

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

conv1 = tf.layers.conv2d(x, 32, [5, 5], 'same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], strides=2)

conv2 = tf.layers.conv2d(pool1, 64, [5, 5], 'same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, [2, 2], strides=2)

dense = tf.layers.dense(pool2, 1024, tf.sigmoid)
y = tf.layers.dense(dense, 1)
y_sigm = tf.sigmoid(y)

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