"""Sprout Challenge: write a controller that gets Sprout, a 28 kg quadruped, down an obstacle course fastest."""
from .controller import Controller
from .sim import Observation, Randomization, Sim
from .track import Track, generate

__all__ = ["Controller", "Observation", "Randomization", "Sim", "Track", "generate"]
__version__ = "0.1.0"
