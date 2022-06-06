from dataclasses import dataclass


@dataclass
class NeuralInput():
    cart_position: float
    cart_velocity: float
    pole_angle: float
    pole_angular_velocity: float
