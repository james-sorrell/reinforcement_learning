import config
import cartpole.q_learning_rbf_sgd as q_learning
import numpy as np
import tensorflow as tf
from tensorflow import keras

class SGDRegressor:
  def __init__(self, D, learning_rate=0.1):
    print("Using Tensorflow SGD Regressor")

    self.lr = learning_rate
    self.w = tf.Variable(tf.random.normal(shape=(D,1)), name='w')
    
    # Multiply and flatten
    @tf.function
    def _forward(X):
      return tf.reshape( tf.matmul(X, self.w), [-1] )

    self.model = keras.Sequential()
    self.model.add(keras.layers.Lambda(self._forward))
    # define optimizer
    self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)

  # Define training function as function definition
  # As above, updates is defined as a function that 
  # can be derived purely from X, Y and the weights
  # that are already stored on the SGD Regressor.
  def partial_fit(self, X, Y):
    with tf.GradientTape() as t:
      delta = Y - self.predict(X)
      cost = tf.reduce_sum(delta*delta)
    grads = t.gradient(cost, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

  # As above, but just to Y_hat stage
  def predict(self, X):
    return self.model(X)

if __name__ == '__main__':
  # Replace q_learning SGD Regressor with Theano SGD
  q_learning.SGDRegressor = SGDRegressor
  q_learning.main()
