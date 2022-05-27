import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import tensorflow as tf
print(tf.__version__)

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
x = tf.random.normal(shape=(1, 10, 100))
y = tf.keras.layers.Conv1D(100, 3)(x)

print(y)
sess = tf.Session()
print(sess)
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(c))
print(sess.run(y))
