"""Zero-forward, training-free observation pruning for coding agents.

The package never imports an LLM client and never performs a model forward.
It compacts tool output before mini-swe-agent formats the next observation.
"""

from zero_forward_pruning.protocol import PruningRequest, PruningResult
from zero_forward_pruning.registry import METHODS, build_pruner

__all__ = ["METHODS", "PruningRequest", "PruningResult", "build_pruner"]
