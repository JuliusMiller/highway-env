import random

import gymnasium as gym

import highway_env
from matplotlib import pyplot as plt
from highway_env.envs.common.action import DiscreteMetaAction

"""
Script for performing one run.
"""

# Register Environments
highway_env.register_highway_envs()

# Choose Environment
#env = gym.make('pedestrian-env-v0', render_mode='rgb_array')
env = gym.make('pedestrian-fixed-landmark-env-v0', render_mode='rgb_array')

# Configure
env.configure({
    "layout": 'Straight',
    "lanes_count": 4,
    "screen_width": 600,
    "screen_height": 600,
    "scaling": 4,
    "duration": 60,
    "action": {
                "type": "ContinuousAction"
    }
})
env.configure({
    "manual_control": False
})
env.reset()


# Set variables for loop
ACTION_TEST = list(DiscreteMetaAction.ACTIONS_ALL.keys())
#ACTION_TEST = list(DiscreteMetaAction.ACTIONS_ALL.values())
FORWARD = [1] * 10 + [0, 2] * 4 + [1] * 80
x_coord = 250
y_coord = -20

done = truncated = False
i = 1

while not done and not truncated:
    #action = env.action_type.actions_indexes['IDLE']  # action always IDLE
    #action = random.choice(ACTION_TEST)  # perform random action
    #action = FORWARD[i]  # perform fixed action
    action = [0, 0]  # Continuos Action

    if i == 10 or i == 11:
        action = [0, 0.79]

    # perform step
    obs, rewards, done, truncated, info = env.step(action)

    # Move goal if necessary
    #env.move_landmark((x_coord, y_coord))
    x_coord += 1
    y_coord = y_coord + 1 if x_coord < 20 or x_coord > 40 else y_coord - 1
    #print(f"step: {i} -> {info}")
    i += 1

    # render + print exit condition
    env.render()
    if done or truncated:
        print(f"Done: {done}, Truncated: {truncated}")
        break
env.close()

# show last frame
plt.imshow(env.render())
plt.show()