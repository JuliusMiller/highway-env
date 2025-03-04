.. _pedestrian_ped_kinematics:

.. py:module::highway_env.pedestrian.kinematics

Kinematics
==================

The vehicles kinematics are represented in the :py:class:`~highway_env.pedestrian.kinematics.Human` class by the *Unicycle Model* :cite:`unicycle`.

.. math::
        \dot{x}&=v\cos(\psi) \\
        \dot{y}&=v\sin(\psi) \\
        \dot{v}&=u_1 \\
        \dot{\psi}&=u_2 \\
where

- :math:`(x, y)` is the vehicle position;
- :math:`v` its forward speed;
- :math:`\psi` its heading;
- :math:`u_1` is the acceleration command;
- :math:`u_2` is the heading command;

These calculations appear in the :py:meth:`~highway_env.vehicle.kinematics.Pedestrian.step` method.

API
***

.. automodule:: highway_env.pedestrian.kinematics
    :members: