import gymnasium as gym
from gym.wrappers import RecordVideo

import highway_env
from matplotlib import pyplot as plt
from highway_env.envs.common.action import DiscreteMetaAction, ContinuousAction

"""
Script to record one run.
Detailed Information available in normal run.
"""


highway_env.register_highway_envs()

env = gym.make('pedestrian-env-v0', render_mode='rgb_array')
#env = gym.make('pedestrian-moving-landmark-env-v0', render_mode='rgb_array')
#env = gym.make('pedestrian-fixed-landmark-env-v0', render_mode='rgb_array')

env.configure({
    "layout": 'Oncoming',
    "lanes_count": 4,
    "screen_width": 1100, #1800 #1100
    "screen_height": 600, #500 #300
    "duration": 20,
    "action": {"type": "ContinuousAction"},
    "scaling": 7, # 7 #9
    "vehicles_count": 5,
    "centering_position": [0.5, 0.5],
    "pedestrian_coordinates": (70, 2),
    "athlete": True
})
env.configure({"observation": {
        "type": "Kinematics",
        "vehicles_count": 6,
        "features": ["presence", "vx", "vy"],
        "features_range": {
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "seebehind": True,
        "order": "sorted"
    }})

env.configure({
    "manual_control": True,
})
env.configure({
    "offscreen_rendering": False
})
env.reset()

env = RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)
env.unwrapped.set_record_video_wrapper(env)


ACTION_TEST = list(DiscreteMetaAction.ACTIONS_ALL.keys())
#ACTION_TEST = list(DiscreteMetaAction.ACTIONS_ALL.values())
FORWARD = [1] * 10 + [2] * 2 + [1] * 10 + [0] * 5 + [1] * 10 + [2] * 3 + [1] * 80
FORWARD = [1] * 5 + [0] * 3 + [3] * 2 + [2] * 3 + [1] * 100
FORWARD = [1] * 5 + [3] * 5 + [1] * 10 + [0] * 2 + [1] * 2 + 4 * [2] + [1] * 2 + [0] * 2 + [1] * 100
x_coord = 595
y_coord = 10
#for i in range(50):
done, truncated = False, False
i = 0
car = []
while not done or not truncated:
    #action = env.action_type.actions_indexes['IDLE']
    #action = random.choice(ACTION_TEST)
    #action = FORWARD[i]
    #print(action)
    action = [0, 0]
    i += 1
    obs, rewards, done, truncated, info = env.step(action)
    #print(obs)
    car.append(obs)
    #env.move_landmark((x_coord, y_coord))
    if i % 5 == 0:
        y_coord = 10 if i % 2 == 0 else 15
    else:
        y_coord = y_coord
    x_coord = x_coord - 3

    #x_coord += 1
    #y_coord = y_coord + 1 if x_coord < 20 or x_coord > 40 else y_coord - 1
    '''if i == 20:
        x_coord = 0
        y_coord = 8
    if i == 40:
        x_coord = 0
        y_coord = -12
    if i == 60:
        x_coord = 20
        y_coord = -12
    if i == 80:
        x_coord = 20
        y_coord = 8
    if i == 85:
        x_coord = 20
        y_coord = -12'''
    #print(f"step: {i} -> {info}")
    #env.step(env.action_space.sample())
    #env.render()
    if done or truncated:
        print(f"{done}, {truncated}")
        break
env.close()


# Function to plot some observations.
def plot():
    x = range(len(car))
    ego = [row[0] for row in car]
    car1 = [row[1] for row in car]  # vx, vy
    car2 = [row[2] for row in car]  # vx, vy
    car3 = [row[3] for row in car]  # vx, vy
    car4 = [row[4] for row in car]  # vx, vy
    car5 = [row[5] for row in car]  # vx, vy

    egox = [row[1] for row in ego]
    car1x = [row[1] for row in car1]
    car2x = [row[1] for row in car2]
    car3x = [row[1] for row in car3]
    car4x = [row[1] for row in car4]
    car5x = [row[1] for row in car5]

    egoy = [row[2] for row in ego]
    car1y = [row[2] for row in car1]
    car2y = [row[2] for row in car2]
    car3y = [row[2] for row in car3]
    car4y = [row[2] for row in car4]
    car5y = [row[1] for row in car5]

    plt.figure(1)
    plt.subplot(211)
    plt.plot(x, egox, 'r', x, car1x, 'b', x, car2x, 'g', x, car3x, 'y', x, car4x, 'c', x, car5x, 'm')
    plt.xlabel('Zeitschritt')
    plt.ylabel('v in x-Richtung')
    plt.subplot(212)
    plt.plot(x, egoy, 'r', x, car1y, 'b', x, car2y, 'g', x, car3y, 'y', x, car4y, 'c', x, car5y, 'm')
    plt.xlabel('Zeitschritt')
    plt.ylabel('v in y-Richtung')
    plt.suptitle('Straight Enviroment')
    plt.show()