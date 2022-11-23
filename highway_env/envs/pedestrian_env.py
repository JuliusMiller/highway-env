from typing import Dict, Text

import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.pedestrian.controller import MDPHuman, ControlledHuman
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.pedestrian.kinematics import Human
from highway_env.vehicle.objects import Obstacle


class PedestrianEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "vehicles_count": 1,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        #self.road = self.create_oval()
        self.road = self.create_straight()

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        self.controlled_vehicles = []

        vehicle = Vehicle(self.road, self.lane.position(0.0, 0), self.lane.heading_at(0.0), speed=3)  # Vehicle at Position 0
        vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
        self.road.vehicles.append(vehicle)

        for _ in range(self.config["vehicles_count"] - 1):
            vehicle = other_vehicles_type.create_random(self.road, speed=0, spacing=1 / self.config["vehicles_density"])
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        #human = Human(self.road, self.lane.position(50, 5), self.lane.heading_at(0), speed=0)
        human = MDPHuman(self.road, self.lane.position(50, 5), self.lane.heading_at(0), speed=0)
        #human = ControlledHuman(self.road, self.lane.position(50, 5), self.lane.heading_at(0), speed=1)

        self.controlled_vehicles.append(human)
        self.road.vehicles.append(human)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return self.vehicle.crashed or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.time >= self.config["duration"]

    def create_straight(self):
        net = RoadNetwork()
        lane = StraightLane([0, 0], [300, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5,
                            speed_limit=100)
        self.lane = lane
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b",
                     StraightLane([0, 5], [300, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                  speed_limit=100))

        return Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def create_oval(self):
        net = RoadNetwork()

        # Oval
        # Set Speed Limits for Road Sections - Straight, Turn 1, Straight, Turn 2
        speedlimits = [10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane([42, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5,
                            speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b",
                     StraightLane([42, 5], [100, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                  speed_limit=speedlimits[1]))
        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-90), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1 + 5, np.deg2rad(90), np.deg2rad(-90), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 3 - Straight Line 2
        net.add_lane("c", "d",
                     StraightLane([42, -40], [100, -40], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                  speed_limit=speedlimits[3]))
        net.add_lane("c", "d",
                     StraightLane([42, -45], [100, -45], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                  speed_limit=speedlimits[3]))

        # 4 - Circular Arc #2
        center1 = [42, -20]
        radii1 = 20
        net.add_lane("d", "a",
                     CircularLane(center1, radii1, np.deg2rad(-90), np.deg2rad(-270), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("d", "a",
                     CircularLane(center1, radii1 + 5, np.deg2rad(-90), np.deg2rad(-270), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        return Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])


    register(
        id='pedestrian-env-v0',
        entry_point='highway_env.envs:PedestrianEnv',
    )