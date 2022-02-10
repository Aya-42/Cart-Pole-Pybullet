# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:50:00 2021

@author: Aya Arbiat
"""

#Install Dependencies__________________________________________________________
import gym 
import random
import pybullet_envs

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

#Set up Environment with OpenAI Gym____________________________________________
env = gym.make("CartPoleBulletEnv-v1")
env.render(mode="human")
states = env.observation_space.shape[0]
actions=env.action_space.n

#Test Random Environment with OpenAI Gym_______________________________________
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = random.choice([0,1])
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))

#Create a Deep Learning Model with Keras_______________________________________
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(states, actions)
model.summary()

#Build Agent with Keras-RL_____________________________________________________
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

_ = dqn.test(env, nb_episodes=15, visualize=True)






