# Memory-Dynamics-Simulation
Python simulation of interacting memory dynamics using differential equations and object-oriented modeling to predict behavioral outcomes.
# 🧠 Memory Dynamics Simulation

Simulate and visualize *interacting memory systems* with linear dynamics in Python. Memories evolve based on self-dynamics (α), interactions (β), and optional biases.

*Mathematical Model:*
$$
\dM_i/dt = \α_i * M_i + Σ_j β_ij * M_j + b_i, for i = 1,...,n
$$
*Features:* Linear & abstract models, interactive input, memory strength evolution, pairwise phase portraits, behavior prediction.

*Usage:* python memory_dynamics.py → enter parameters, initial strengths, simulation time & threshold.

*Output:* Time evolution plots, dominant memory summary, pairwise phase portraits.
Requirements: numpy, scipy, matplotlib.

