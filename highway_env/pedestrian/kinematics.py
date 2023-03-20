import random
from typing import Union, Optional, Tuple, List
import numpy as np
import copy
from collections import deque

from highway_env import utils
from highway_env.road.road import Road, LaneIndex
from highway_env.vehicle.objects import RoadObject, Obstacle, Landmark
from highway_env.utils import Vector


class Human(RoadObject):
    """
    A moving human on a road, and its kinematics.

    The human is represented by a dynamical system: a modified unicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    RADIUS = 0.5
    '''Radius of Human [m]'''

    LENGTH = 2 * RADIUS
    WIDTH = 2 * RADIUS
    MAX_SPEED = 3.
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = -3.
    """ Minimum reachable speed [m/s] """
    DEFAULT_INITIAL_SPEEDS = [1, 1.5]
    """ Range for random initial speeds [m/s] """

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = random.uniform(DEFAULT_INITIAL_SPEEDS[0], DEFAULT_INITIAL_SPEEDS[1])):
        super().__init__(road, position, heading, speed)
        self.action = {'steering': 0.0, 'acceleration': 0.0}
        self.crashed = False
        self.impact = None
        self.color = (50, 150, 0)

    @classmethod
    def create_human(cls, road: Road,
                      speed: float = None,
                      lane_from: Optional[str] = None,
                      lane_to: Optional[str] = None,
                      lane_id: Optional[int] = None) \
            -> "Human":
        """
        Create a random human on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the human is located
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A human with random position and/or speed
        """

        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            speed = road.np_random.uniform(Human.DEFAULT_INITIAL_SPEEDS[0], Human.DEFAULT_INITIAL_SPEEDS[1])
        default_spacing = 12 + 1.0 * speed
        offset = default_spacing * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles]) \
            if len(road.vehicles) else 3 * offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        h = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return h

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt: float) -> None:
        """
        Propagate the human state given its actions.

        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions()
        v = self.speed * np.array([np.cos(self.heading),
                                   np.sin(self.heading)])
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        steering = self.action['steering']
        acceleration = self.action['acceleration']
        self.heading += steering * dt
        self.speed += acceleration * dt

    def clip_actions(self) -> None:
        if self.crashed:
            self.action['steering'] = 0.0
            self.action['acceleration'] = -1.0*self.speed
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        if self.speed > self.MAX_SPEED:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))
        elif self.speed < self.MIN_SPEED:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0 * (self.MIN_SPEED - self.speed))


class Athlete(Human):
    """
    An extension for the Human class.
    """

    MAX_SPEED = 10.
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = -7
    """ Minimum reachable speed [m/s] """