"""Custom exception types for the phyla library."""

from __future__ import annotations


class PhylaValidationError(ValueError):
    """Raised when a Phyloseq object fails component-consistency validation.

    R reference: phyloseq raises stop() with similar messages; this is the
    Python equivalent, subclassing ValueError so it can be caught generically.
    """
