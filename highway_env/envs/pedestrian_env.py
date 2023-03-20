import random
from typing import Dict, Text, Optional

import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs import AbstractEnv
from highway_env.envs.common.abstract import Observation
from highway_env.envs.common.action import Action, ContinuousAction, DiscreteMetaAction
from highway_env.pedestrian.controller import ControlledHuman, FollowHuman, FollowAthlete, ControlledAthlete
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.pedestrian.kinematics import Human, Athlete
from highway_env.vehicle.objects import Obstacle, Landmark


class PedestrianEnv(AbstractEnv):
    """
    Environment for pedestrian movement.

    A pedestrian is spawned and moving around interacting with vehicles
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "DiscreteMetaAction",
                "pedestrian": "True"
            },
            "vehicles_count": 5,
            "controlled_humans": 1,
            "lanes_count": 2,
            "duration": 60,  # [s]
            "vehicles_density": 1,
            "offroad_terminal": False,
            "layout": 'Straight',
            "pedestrian_speed": 1,  # [m/s]
            "pedestrian_coordinates": (20, -10),
            "pedestrian_heading": 0,
            "athlete": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self._create_human()

    def _create_road(self) -> None:
        """Create a road according to the chosen layout."""
        layout = self.config['layout']
        if layout == 'Straight':
            self.road = self.create_straight()
        elif layout == 'Oval':
            self.road = self.create_oval()
        elif layout == 'Intersection':
            self.road = self.create_intersection()
        elif layout == 'Oncoming':
            self.road = self.create_oncoming()
        else:
            raise NotImplementedError(f"{layout} not implemented")

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        speeds = [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]
        if self.config['vehicles_count'] > 0:
            vehicle = Vehicle(self.road, self.lane.position(0.0, 0), self.lane.heading_at(0.0), speed=random.sample(speeds, 1)[0])   # Vehicle at Position 0
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.road.vehicles.append(vehicle)

        for i in range(self.config["vehicles_count"] - 1):
            if self.config['layout'] == 'Oncoming':
                if i % 2 == 0:
                    vehicle = other_vehicles_type(self.road, position=self.road.network.get_lane(("b", "a", 0)).position(10 + i * 10, 0),
                              heading=self.road.network.get_lane(("b", "a", 0)).heading_at(10 + 10*i),
                              speed=random.sample(speeds, 1)[0])
                else:
                    vehicle = other_vehicles_type(self.road, position=self.road.network.get_lane(("a", "b", 0)).position(10 + i * 10, 0),
                              heading=self.road.network.get_lane(("a", "b", 0)).heading_at(10 + 10*i),
                              speed=random.sample(speeds, 1)[0])
            else:
                vehicle = other_vehicles_type.create_random(self.road, speed=random.sample(speeds, 1)[0], spacing=1 / self.config["vehicles_density"]) # random.sample(speeds, 1)[0]
            vehicle.randomize_behavior()

            self.road.vehicles.append(vehicle)

    def _create_human(self) -> None:
        """
        Create the pedestrian according to the chosen ActionType.
        """
        self.controlled_vehicles = []
        # human = Human(self.road, self.lane.position(50, 5), self.lane.heading_at(0), speed=0)
        x, y = self.config['pedestrian_coordinates']
        speed = self.config['pedestrian_speed']
        heading = self.config['pedestrian_heading']
        athlete = self.config['athlete']
        if isinstance(self.action_type, ContinuousAction):
            human = Human(self.road, self.lane.position(x, y), heading=heading, speed=speed)
        elif isinstance(self.action_type, DiscreteMetaAction):
            human = ControlledHuman(self.road, self.lane.position(x, y), heading=heading, speed=speed)
        elif isinstance(self.action_type, ContinuousAction) and athlete:
            human = Athlete(self.road, self.lane.position(x, y), heading=heading, speed=speed)
        elif isinstance(self.action_type, DiscreteMetaAction) and athlete:
            human = ControlledAthlete(self.road, self.lane.position(x, y), heading=heading, speed=speed)

        self.controlled_vehicles.append(human)
        self.road.vehicles.append(human)

    def _reward(self, action: Action) -> float:
        """
        :param action: the last action performed
        :return: the corresponding reward
        """
        return 0.0

    def _rewards(self, action: Action) -> Dict[Text, float]:
        return {
            "collision_reward": 0.0
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return self.vehicle.crashed or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.time >= self.config["duration"]

    def create_straight(self):
        """Create Straight layout"""
        net = RoadNetwork()
        width = 5
        lenght = 700
        lane = StraightLane([0, 0], [lenght, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED if self.config["lanes_count"] > 1 else LineType.CONTINUOUS), width=width,
                            speed_limit=20)
        self.lane = lane
        net.add_lane("a", "b", lane)
        i = 1
        while i < self.config["lanes_count"]:
            net.add_lane("a", "b",
                         StraightLane([0, i * width], [lenght, i * width], line_types=(LineType.STRIPED, LineType.STRIPED if i + 1 < self.config["lanes_count"] else LineType.CONTINUOUS), width=width,
                                      speed_limit=20))
            i += 1

        return Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def create_oval(self):
        """Create Oval layout"""
        net = RoadNetwork()

        # Oval
        # Set Speed Limits for Road Sections - Straight, Turn 1, Straight, Turn 2
        speedlimits = [20, 10, 30, 10]

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

    def create_intersection(self) -> Road:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            self.lane = StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10)
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4), self.lane)
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        return  RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def create_oncoming(self, length=150):
        """
        Make a road composed of a two-way road.

        :return: the road
        """
        net = RoadNetwork()
        self.lane = StraightLane([0, 0], [length, 0], line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED))
        # Lanes
        net.add_lane("a", "b", self.lane)
        '''net.add_lane("a", "b", StraightLane([0, StraightLane.DEFAULT_WIDTH], [length, StraightLane.DEFAULT_WIDTH],
                                            line_types=(LineType.NONE, LineType.CONTINUOUS_LINE)))'''
        net.add_lane("b", "a", StraightLane([length, StraightLane.DEFAULT_WIDTH], [0, StraightLane.DEFAULT_WIDTH],
                                            line_types=(LineType.CONTINUOUS_LINE, LineType.NONE)))

        return Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
            "heading": self.vehicle.heading,
        }
        return info


class PedestrianFixedLandmark(PedestrianEnv):
    """
    Environment for fixed pedestrian movement.

    A pedestrian is spawned and walks towards fixed goals.
    """

    def _create_road(self) -> None:
        """
        Creates road + sets all goals.
        """
        super()._create_road()
        self.goal = Landmark(self.road, self.lane.position(20, 10), heading=self.lane.heading)
        self.road.objects.append(self.goal)
        self.road.objects.append(
            Landmark(self.road, self.lane.position(20, -5), heading=self.lane.heading))
        self.road.objects.append(
            Landmark(self.road, self.lane.position(22, -5), heading=self.lane.heading))
        self.road.objects.append(
            Landmark(self.road, self.lane.position(22, 10), heading=self.lane.heading))
        '''self.road.objects.append(
            Landmark(self.road, self.lane.position(self.lane.length / 3, 20), heading=self.lane.heading))
        self.road.objects.append(
            Landmark(self.road, self.lane.position(self.lane.length / 2, -5), heading=self.lane.heading))
        self.road.objects.append(
            Landmark(self.road, self.lane.position(self.lane.length / 2, 5), heading=self.lane.heading))'''

    def _create_human(self) -> None:
        """
        Creates only one FollowHuman capable of walking towards goals.
        """
        self.controlled_vehicles = []
        x, y = self.config['pedestrian_coordinates']
        speed = self.config['pedestrian_speed']
        heading = self.config['pedestrian_heading']
        athlete = self.config['athlete']
        if athlete:
            human = FollowAthlete(self.road, self.lane.position(x, y), heading=heading, speed=speed)
        else:
            human = FollowHuman(self.road, self.lane.position(x, y), heading=heading, speed=speed)

        self.controlled_vehicles.append(human)
        self.road.vehicles.append(human)


class PedestrianMovingLandmark(PedestrianEnv):
    """
    Environment for variable pedestrian movement.

    A pedestrian is spawned and walks towards moving goals.
    """
    def _create_road(self) -> None:
        """
        Creates road + sets first goal.
        """
        super()._create_road()
        self.moving_goal = Landmark(self.road, self.lane.position(0, 20), heading=self.lane.heading)
        self.road.objects.append(self.moving_goal)

    def _create_human(self) -> None:
        """
        Creates only one FollowHuman capable of walking towards goals.
        """
        self.controlled_vehicles = []
        x, y = self.config['pedestrian_coordinates']
        speed = self.config['pedestrian_speed']
        heading = self.config['pedestrian_heading']
        athlete = self.config['athlete']
        if athlete:
            human = FollowAthlete(self.road, self.lane.position(x, y), heading=heading, speed=speed)
        else:
            human = FollowHuman(self.road, self.lane.position(x, y), heading=heading, speed=speed)

        self.controlled_vehicles.append(human)
        self.road.vehicles.append(human)

    def move_landmark(self, pos):
        """Moves the goal to a new position"""
        self.road.objects.remove(self.moving_goal)
        x, y = pos
        self.moving_goal = Landmark(self.road, self.lane.position(x, y), heading=self.lane.heading)
        self.road.objects.append(self.moving_goal)
