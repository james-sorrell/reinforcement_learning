import config
import cartpole.q_learning_rbf_sgd as q_learning
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


class SGDRegressor:
  def __init__(self, D, learning_rate=0.1):
    print("Using Tensorflow SGD Regressor")

    self.lr = learning_rate
    self.w = tf.Variable(tf.random.normal(shape=(D,1)), name='w')
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')
    # Multiply and flatten
    Y_hat = tf.reshape( tf.matmul(self.X, self.w), [-1] )
    delta = self.Y - Y_hat
    # Sums along the vector, reducing to scalar
    # Same as dot product of delta
    cost = tf.reduce_sum(delta*delta)
    self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(cost)
    self.predict_op = Y_hat

    # Initialise tf instance
    init = tf.global_variables_initializer()
    self.session = tf.InteractiveSession()
    self.session.run(init)

  # Define training function as function definition
  # As above, updates is defined as a function that 
  # can be derived purely from X, Y and the weights
  # that are already stored on the SGD Regressor.
  def partial_fit(self, X, Y):
    self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})
  # As above, but just to Y_hat stage
  def predict(self, X):
    return self.session.run(self.predict_op, feed_dict={self.X: X})

if __name__ == '__main__':
  # Replace q_learning SGD Regressor with Theano SGD
  q_learning.SGDRegressor = SGDRegressor
  q_learning.main()
