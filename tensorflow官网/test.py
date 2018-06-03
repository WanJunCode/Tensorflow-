import tensorflow as tf

labels=[0,0,1,0,0,0,0,0,0,0]

# 获得张量的大小
batch_size = tf.size(labels)

# inserts a dimension of 1 into a tensor's shape.
# [10] --> [10,1]
labels2 = tf.expand_dims(input=labels, axis=1)
indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(batch_size))
    print(sess.run(labels2))
    print(sess.run(indices))