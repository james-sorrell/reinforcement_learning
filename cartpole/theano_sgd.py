import config
import theano
import cartpole.q_learning_rbf_sgd as q_learning
import numpy as np
import theano.tensor as T


class SGDRegressor:
  def __init__(self, D, learning_rate=0.1):
    print("Using Theano SGD Regressor")
    # Xavier Initialisation
    # https://stats.stackexchange.com/questions/326710/why-is-weight-initialized-as-1-sqrt-of-hidden-nodes-in-neural-networks
    w = np.random.randn(D) / np.sqrt(D)
    # Variable with Storage that is shared between functions that it appears in. 
    self.w = theano.shared(w)
    self.lr = learning_rate

    X = T.matrix('X')
    Y = T.vector('Y')
    Y_hat = X.dot(self.w)
    delta = Y - Y_hat
    # Squared Loss
    cost = delta.dot(delta)
    # Dcost / Dw
    grad = T.grad(cost, self.w)
    updates = [(self.w, self.w - self.lr*grad)]

    # Define training function as function definition
    # As above, updates is defined as a function that 
    # can be derived purely from X, Y and the weights
    # that are already stored on the SGD Regressor.
    self.train_op = theano.function(
      inputs=[X, Y],
      updates=updates,
    )
    # As above, but just to Y_hat stage
    self.predict_op = theano.function(
      inputs=[X],
      outputs=Y_hat,
    )

  # API to access internal SGD functions

  def partial_fit(self, X, Y):
    self.train_op(X, Y)

  def predict(self, X):
    return self.predict_op(X)


if __name__ == '__main__':
  # Replace q_learning SGD Regressor with Theano SGD
  q_learning.SGDRegressor = SGDRegressor
  q_learning.main()
