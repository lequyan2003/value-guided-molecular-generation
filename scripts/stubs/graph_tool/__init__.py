"""Empty stand-in for graph-tool (fallback only). DiGress's main.py does ``import graph_tool as gt`` on line 1, but graph-tool is
used for real only by the SBM/planar metrics (``analysis/spectre_utils.py``), which the QM9 path never imports.
Use it with ``PYTHONPATH=scripts/stubs`` only if the conda install of graph-tool fails, and say so in the report."""
