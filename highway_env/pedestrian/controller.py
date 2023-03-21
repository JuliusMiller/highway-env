from typing import List, Tuple, Union, Optional

import numpy as np
import copy
from highway_env import utils
from highway_env.road.road import Road, LaneIndex, Route
from highway_env.utils import Vector
from highway_env.pedestrian.kinematics import Human, Athlete
from highway_env.vehicle.objects import Landmark


class ControlledHuman(Human):
    """
    A Human piloted by a low-level controller.

    - The longitudinal controller is a speed controller
    - The lateral controller is a heading controller
    """

    KP_ACC = 1.5
    KP_ANGLE = 1.5
    MAX_STEERING_ANGLE = np.pi / 2  # [rad]
    ACC_MAX = 1.0  # [m/s2]
    """Maximum acceleration."""
    DELTA_SPEED = 1 / 3.6  # [m/s]
    DELTA_ANGLE = np.pi / 8

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_speed: float = None
                 ):
        super().__init__(road, position, heading, speed)
        self.target_speed = target_speed or self.speed
        self.target_angle = heading

    @classmethod
    def create_from(cls, human: "ControlledHuman") -> "ControlledHuman":
        """
        Create a new human from an existing one.

        The human dynamics and target dynamics are copied, other properties are default.

        :param human: a Human
        :return: a new human at the same dynamical state
        """
        h = cls(human.road, human.position, heading=human.heading, speed=human.speed,
                target_speed=human.target_speed)
        return h

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and heading;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "RIGHT_TURN":
            self.target_angle += self.DELTA_ANGLE
        elif action == "LEFT_TURN":
            self.target_angle -= self.DELTA_ANGLE

        if np.abs(self.target_angle) < 1e-4:
            self.target_angle = 0.0
        if np.abs(self.target_speed) < 1e-4:
            self.target_speed = 0.0
        action = {"steering": self.steering_control(self.target_angle),
                  "acceleration": self.speed_control(self.target_speed)}
        super().act(action)

    def safe_angle(self, angle):
        """
        :param angle: angle
        :return: angle in correct quadrant
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    def steering_control(self, steering_angle: float) -> float:
        """
        Control the heading of the vehicle.

        Using a simple proportional controller.

        :param steering_angle: new angle
        :return: a steering wheel angle command [rad]
        """
        e = steering_angle - self.heading
        steering_angle = self.KP_ANGLE * self.safe_angle(e)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        acceleration = self.KP_ACC * (target_speed - self.speed)
        acceleration = np.clip(acceleration, -self.ACC_MAX, self.ACC_MAX)
        return float(acceleration)


class FollowHuman(ControlledHuman):
    """
    A Human capable of walking towards some goals.
    """
    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_speed: float = None):
        super().__init__(road, position, heading, speed, target_speed)
        self.goals = self.road.objects  # [lm for lm in self.road.objects if isinstance(lm, Landmark)], better but wrong reference in moveLandmark

    @classmethod
    def create_from(cls, human: "ControlledHuman") -> "FollowHuman":
        """
        Create a new human from an existing one.

        The human dynamics and target dynamics are copied, other properties are default.

        :param human: a human
        :return: a new human at the same dynamical state
        """
        h = cls(human.road, human.position, heading=human.heading, speed=human.speed,
                target_speed=human.target_speed)
        return h

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

            1. Update Goal if necessary
            2. Calculate heading
            3. Calculate speed


        :param action: a high-level action, gets ignored
        """
        for g in self.goals:
            if np.abs(g.position[0] - self.position[0]) < 1 and np.abs(g.position[1] - self.position[1]) < 1:
                self.goals.remove(g)
                self.goals.append(g)
        goal_pos = self.goals[0].position if self.goals else [self.position[0], self.position[1]]

        #Go to goal
        self.target_angle = np.arctan2(goal_pos[1] - self.position[1], goal_pos[0] - self.position[0])

        #Euklidische Distanz
        self.target_speed = np.sqrt(np.power(goal_pos[0] - self.position[0], 2) + np.power(goal_pos[1] - self.position[1], 2)) / 4

        super().act(self)


class ControlledAthlete(Athlete, ControlledHuman):

    """Extension to ControlledHuman"""
    KP_ACC = 2
    KP_ANGLE = 2

    DELTA_SPEED = 1  # [m/s]
    DELTA_ANGLE = np.pi / 6
    ACC_MAX = 4.0  # [m/s2]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_speed: float = None):
        ControlledHuman.__init__(self, road, position, heading, speed, target_speed)


class FollowAthlete(ControlledAthlete, FollowHuman):
    """Extension to FollowHuman"""
    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_speed: float = None):
        FollowHuman.__init__(self, road, position, heading, speed, target_speed)
