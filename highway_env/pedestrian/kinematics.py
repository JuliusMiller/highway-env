from typing import Union, Optional, Tuple, List
import numpy as np
import copy
from collections import deque

from highway_env import utils
from highway_env.road.road import Road, LaneIndex
from highway_env.vehicle.objects import RoadObject, Obstacle, Landmark
from highway_env.utils import Vector



#TODO: Unterschied zwischen Gehen und Rennen

class Human(RoadObject):
    RADIUS = 1
    '''Radius of Human [m]'''
    DEFAULT_INITIAL_SPEEDS = [1, 2]
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 20.
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = -2.
    """ Minimum reachable speed [m/s] """

    HISTORY_SIZE = 30
    """ Length of the vehicle state history, for trajectory display"""

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 predition_type: str = 'constant_steering'):
        super().__init__(road, position, heading, speed)
        self.prediction_type = predition_type
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.impact = None
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)
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

        :param road: the road where the vehicle is driving
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
        delta_f = self.action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action['acceleration'] * dt
        self.on_state_update()

    def clip_actions(self) -> None:
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.speed
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        if self.speed > self.MAX_SPEED:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))
        elif self.speed < self.MIN_SPEED:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0 * (self.MIN_SPEED - self.speed))

    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)