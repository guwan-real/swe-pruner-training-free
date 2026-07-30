"""Training-free, post-action pruning for coding-agent observations.

This package is intentionally independent from ``tf_pruning``.  The older
package implements pre-action ``query + observation`` rankers.  This package
scores an action that the agent has already generated from the full
observation and only compacts that observation for future turns.
"""

from posterior_pruning.protocol import PosteriorPruningRequest, PosteriorPruningResult

__all__ = ["PosteriorPruningRequest", "PosteriorPruningResult"]
