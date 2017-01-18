import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in Numpy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3


# print(x_data, y_data, sep='\n')

# Try to find values for W and b that compute y_data = W * x_data + b.
# We know that W should be 0.1 and b should be 0.3, but TensorFlow will figure that out.
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# print(W, b, y, sep='\n')

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# print(loss, optimizer, train, sep='\n')

# Before starting, initialize variable.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]