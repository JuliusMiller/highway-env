.. _pedestrian_ped_controller:

Control
========

The :py:class:`~highway_env.pedestrian.controller.ControlledHuman` class implements a low-level controller on top of a :py:class:`~highway_env.pedestrian.kinematics.Human`, allowing to track a given target speed and follow a target heading.
The controls are computed when calling the :py:meth:`~highway_env.pedestrian.controller.ControlledHuman.act` method.

Longitudinal controller
-----------------------

The longitudinal controller is a simple proportional controller:

.. math::
    a = K_p(v_t - v),

where

- :math:`a` is the human acceleration;
- :math:`v` is the human velocity;
- :math:`v_t` is the target velocity;
- :math:`K_p` is the controller proportional gain, implemented as :py:attr:`~highway_env.pedestrian.controller.ControlledHuman.KP_ACC`.

It is implemented in the :py:meth:`~highway_env.pedestrian.controller.ControlledHuman.speed_control` method.

Lateral controller
-----------------------

The lateral controller is a simple proportional controller:

.. math::
    \dot{\phi} &= K_p*\arctan2\left(\frac{\sin(e)}{\cos(e)}\right),\\
    e &= (\phi_t - \phi)

where

- :math:`\dot{\phi}` is the human angular velocity;
- :math:`\phi` is the human heading;
- :math:`\phi_t` is the target heading;
- :math:`K_p` is the controller proportional gain, implemented as :py:attr:`~highway_env.pedestrian.controller.ControlledHuman.KP_ANGLE`.

It is implemented in the :py:meth:`~highway_env.pedestrian.controller.ControlledHuman.heading_control` method.

API
----

.. automodule:: highway_env.pedestrian.controller
    :members:

