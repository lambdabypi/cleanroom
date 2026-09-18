"""Cleanroom: a web-to-clean-dataset ETL agent that writes its own extractors.

Each attempt is scored by running the generated code in a sandbox and validating
the rows it produces, so quality claims are measured rather than asserted. The
learning machinery (contextual Thompson sampling over extraction strategies) is
present and converges in simulation; a controlled run found no significant
advantage over random selection on live pages, and was underpowered to settle it
either way. See runs/README.md before quoting any learning result.
"""

__version__ = "0.1.0"
