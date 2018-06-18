# TensorFlow laden
import tensorflow as tf

# TensorFlow Session definieren

sess = tf.Session()

# Zwei Konstanten

x = tf.constant(3, dtype=tf.int8)
y = tf.constant(2, dtype=tf.int8)


# Eine Multiplikation
z = tf.multiply(x, y)

print sess.run(z)
