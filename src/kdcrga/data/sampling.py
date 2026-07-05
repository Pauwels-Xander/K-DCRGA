"""Neighbour sampling for memory-bounded training on ~8 GB VRAM.

Configures PyG ``NeighborLoader`` fan-out so subgraph batches fit on an RTX 4070.
Filled in stage v1.
"""
