"""Training-free delayed compaction for coding-agent prompt history.

This package deliberately does not replace a freshly produced tool observation.
It records that observation verbatim, waits for the agent's normal next action,
then uses that action as a posterior relevance signal when rendering older
history for later model calls.
"""

from posterior_history_pruning.protocol import PosteriorHistoryConfig

__all__ = ["PosteriorHistoryConfig"]
