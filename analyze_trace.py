"""Compatibility wrapper for the Trace Analyzer CLI and public helpers."""

from trace_analyzer.core import *  # noqa: F401,F403
from trace_analyzer.core import main


if __name__ == "__main__":
    main()
