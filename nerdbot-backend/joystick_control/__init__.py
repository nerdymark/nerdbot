"""
Joystick Control Module for NerdBot

This module provides joystick/gamepad control capabilities for the robot,
translating USB controller inputs into motor and servo commands.
"""

from .joystick_service import JoystickService

__all__ = ['JoystickService']