import tensorflow as tf



with tf.variable_scope("foo",initializer=tf.constant_initializer(0.4)):
    v = tf.get_variable("v",[1])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(v.eval())