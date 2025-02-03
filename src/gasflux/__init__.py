"""Init file for gasflux package."""

from importlib.metadata import version

__version__ = version("gasflux")

from . import (
    background,
    cli,
    gas,
    interpolation,
    ml,
    plotting,
    pre_processing,
    processing,
    processing_pipelines,
    reporting,
)

__all__ = [
    "background",
    "cli",
    "gas",
    "interpolation",
    "ml",
    "plotting",
    "pre_processing",
    "processing",
    "processing_pipelines",
    "reporting",
]
