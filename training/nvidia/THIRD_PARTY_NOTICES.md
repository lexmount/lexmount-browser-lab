# Third-party notice

`vendor/nemo-rl-v0.6.0-ray.sub` is copied from NVIDIA NeMo RL `v0.6.0`:

- Source: `https://github.com/NVIDIA-NeMo/RL/blob/v0.6.0/ray.sub`
- Upstream SHA-256: `b7bff520ca4416e584fca4dbb6033943d1985ac88f4a55d03f0575df12f11943`
- License: Apache-2.0, retained in the file header.

Local modifications: two bare `env` diagnostic calls were removed, and trailing
whitespace was normalized. The former prevents credentials loaded for rollout
workers from being printed to Slurm logs; the latter satisfies repository
whitespace checks. No Ray topology or NeMo training behavior was changed.
