.. _environments_pedestrian:

.. currentmodule:: highway_env.envs.pedestrian_env

Pedestrian
**********

In this environment a pedestrian can walk on the road and interact with vehicles.


Usage
==========

.. code-block:: python

    env = gym.make('pedestrian-env-v0', render_mode='rgb_array')


Default configuration
=====================

.. code-block:: python

    {
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
        }

More specifically, it is defined in:

.. automethod:: PedestrianEnv.default_config

Different variants
=====================

Environment with fixed goals

.. code-block:: python

    env = gym.make('pedestrian-fixed-landmark-env-v0', render_mode='rgb_array')


Environment with moving goals

.. code-block:: python

    env = gym.make('pedestrian-moving-landmark-env-v0', render_mode='rgb_array')

API
=====

.. autoclass:: PedestrianEnv
    :members:
