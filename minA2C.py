"""
A minimal Advantage Actor Critic Implementation

Usage:
python3 minA2C.py
"""

import gym
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras

from collections import deque
import time
import random

RANDOM_SEED = 6
tf.random.set_seed(RANDOM_SEED)

env = gym.make('CartPole-v1')
#env = gym.make('MountainCar-v0')
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

# An episode a full game
train_episodes = 300

def create_actor(state_shape, action_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='softmax', kernel_initializer=init))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def create_critic(state_shape, output_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(output_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def one_hot_encode_action(action, n_actions):
    encoded = np.zeros(n_actions, np.float32)
    encoded[action] = 1
    return encoded

def main():
    actor_checkpoint_path = "training_actor/actor_cp.ckpt"
    critic_checkpoint_path = "training_critic/critic_cp.ckpt"

    actor = create_actor(env.observation_space.shape, env.action_space.n)
    critic = create_critic(env.observation_space.shape, 1)
    if os.path.exists('training_actor'):
        actor.load_weights(actor_checkpoint_path)

        critic.load_weights(critic_checkpoint_path)
    print(actor)
    print(critic)

    # X = states, y = actions
    X = []
    y = []

    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = env.reset()
        done = False
        while not done:
            if True:
                env.render()

            # model dims are (batch, env.observation_space.n)
            observation_reshaped = observation.reshape([1, observation.shape[0]])
            action_probs = actor.predict(observation_reshaped).flatten()
            # Note we're sampling from the prob distribution instead of using argmax
            action = np.random.choice(env.action_space.n, 1, p=action_probs)[0]
            encoded_action = one_hot_encode_action(action, env.action_space.n)

            next_observation, reward, done, info = env.step(action)
            next_observation_reshaped = next_observation.reshape([1, next_observation.shape[0]])

            value_curr = np.asscalar(np.array(critic.predict(observation_reshaped)))
            value_next = np.asscalar(np.array(critic.predict(next_observation_reshaped)))

            # Fit on the current observation
            discount_factor = .7
            TD_target = reward + (1 - done) * discount_factor * value_next
            advantage = critic_target = TD_target - value_curr
            print(np.around(action_probs, 2), np.around(value_next - value_curr, 3), 'Advantage:', np.around(advantage, 2))
            advantage_reshaped = np.vstack([advantage])
            TD_target = np.vstack([TD_target])
            critic.train_on_batch(observation_reshaped, TD_target)
            #critic.fit(observation_reshaped, TD_target, verbose=0)

            gradient = encoded_action - action_probs
            gradient_with_advantage = .0001 * gradient * advantage_reshaped + action_probs
            actor.train_on_batch(observation_reshaped, gradient_with_advantage)
            #actor.fit(observation_reshaped, gradient_with_advantage, verbose=0)
            observation = next_observation
            total_training_rewards += reward

            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                total_training_rewards += 1

                actor.save_weights(actor_checkpoint_path)
                critic.save_weights(critic_checkpoint_path)

    env.close()

if __name__ == '__main__':
    main()
