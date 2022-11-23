import random

import gym
import numpy as np

import highway_env
from matplotlib import pyplot as plt

from highway_env.envs.common.action import DiscreteMetaAction

env = gym.make('pedestrian-env-v0')

env.configure({
    "screen_width": 600,
    "screen_height": 600,
    #"centering_position": [0.5, 0.5],
})
env.configure({
    "manual_control": False
})

env.reset()
'''
ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
'''

ACTION_TEST = list(DiscreteMetaAction.ACTIONS_ALL.keys())

FORWARD = [1] * 30 + [2] * 4 + [1] * 80

for i in range(100):
    #action = env.action_type.actions_indexes["IDLE"]
    #action = random.choice(ACTION_TEST)
    action = FORWARD[i]
    #print(action)
    obs, rewards, done, truncated, info = env.step(action)
    print(info)
    #env.step(env.action_space.sample())
    env.render()
    #if done or truncated:
    #    break

plt.imshow(env.render(mode="rgb_array"))
plt.show()