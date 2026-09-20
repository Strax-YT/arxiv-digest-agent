"""Autonomous arXiv paper digest & QA agent."""

__version__ = "1.0.0"

from .agent import Agent  # noqa: F401
from .config import Settings  # noqa: F401
from .state import AgentState  # noqa: F401

__all__ = ["Agent", "Settings", "AgentState", "__version__"]
