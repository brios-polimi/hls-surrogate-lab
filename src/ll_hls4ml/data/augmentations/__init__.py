"""Composable learning-time graph augmentation hooks."""

from .base import Compose, GraphTransform, register, resolve
from .topology import (  # noqa: F401
    PermuteDestinationsWithinFunction,
    TOPOLOGY_DESTINATION_PERMUTE_VERSION,
)

__all__ = [
    "Compose", "GraphTransform", "register", "resolve",
    "PermuteDestinationsWithinFunction", "TOPOLOGY_DESTINATION_PERMUTE_VERSION",
]
