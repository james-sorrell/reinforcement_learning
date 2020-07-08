# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import numpy as np
import tensorflow as tf
from tensorflow import keras


# def sample(logits):
#     """ Sample from logits distribution helper """
#     noise = tf.random.uniform(tf.shape(logits))
#     return tf.argmax(logits - tf.math.log(-tf.math.log(noise)), 1)

def sharedNetwork(input):
    out = keras.layers.Conv2D(filters=32, kernel_size=8, 
                        strides=4, activation=tf.nn.relu)(input)
    out = keras.layers.Conv2D(filters=64, kernel_size=4,
                        strides=2, activation=tf.nn.relu)(out)
    out = keras.layers.Conv2D(filters=64, kernel_size=3,
                        strides=1, activation=tf.nn.relu)(out)
    out = keras.layers.Flatten()(out)
    return keras.layers.Dense(units=512, activation=tf.nn.relu)(out)

def valueTail(shared):
    return tf.squeeze(keras.layers.Dense(units=1, activation=None)(shared), axis=1)

def policyTail(shared, num_actions):
    return tf.squeeze(keras.layers.Dense(units=num_actions, activation=tf.nn.softmax)(shared))

class SharedModel:

    def __init__(self, obs_shape, num_actions, eps=1e-5, base_clip_epsilon=0.1,
                value_loss_coefficient=1, entropy_loss_coefficient=0.01,
                base_learning_rate=2.5e-4, gradient_max=10.0):
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.base_clip_epsilon = base_clip_epsilon
        self.c_val_loss = value_loss_coefficient
        self.c_ent_loss = entropy_loss_coefficient
        self.base_lr = base_learning_rate
        self.max_grad_norm = gradient_max
        #
        self.X = keras.Input(dtype=tf.float32, shape=(self.obs_shape))
        # Shared
        shared = sharedNetwork(self.X)
        # Policy / Value
        value = valueTail(shared)
        policy = policyTail(shared, num_actions)
        #
        self.network = tf.keras.Model(inputs=self.X, outputs=[value, policy])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.get_lr, epsilon=eps)

    def get_lr(self):
        return self.base_lr*self.alpha

    def train(self, states, action, policy_old, advantage, value_estimate, alpha):
        
        self.alpha = alpha
        # Compute Gradients
        with tf.GradientTape() as t:
            value, policy = self.network(states)
            act_one_hot = keras.backend.one_hot(action, self.num_actions)
            clip_eps = alpha * self.base_clip_epsilon
            # NOTE: Forgot to add axis=1 here, huge bug.
            log_prob_ratio = tf.math.log(tf.math.reduce_sum(policy * act_one_hot, axis=1)) \
                                - tf.math.log(tf.reduce_sum(policy_old * act_one_hot, axis=1))
            prob_ratio = tf.math.exp(log_prob_ratio)
            clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1-clip_eps, 1+clip_eps)
            # NOTE: Needed to be negative, since we want entropy to be big, not small.
            entropy_loss = -tf.math.reduce_sum(-policy * tf.math.log(policy), axis=1)
            # NOTE: Huge bug, needed to maximize the minimum, not minimize it.  Without the minus sign, is not a loss.
            clip_loss = -tf.math.minimum(prob_ratio * advantage, clipped_prob_ratio * advantage)
            # NOTE: Need a better name than value estimate, they are both value estimtates
            value_loss = tf.math.square(value - value_estimate)
            total_loss = tf.math.reduce_mean(clip_loss + self.c_val_loss * value_loss + self.c_ent_loss * entropy_loss)
            # NOTE: Took epsilon from OpenAI Baselines ppo2.
        grads = t.gradient(total_loss, self.network.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))
        return total_loss #, value_loss, clip_loss, entropy_loss   