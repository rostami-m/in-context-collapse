# Local runner support

`storage.py` performs a local output-directory writability check using a unique
temporary probe. Historical Docker, Slurm, SSH, shard, and synchronization
launchers are excluded from this code-only release because they targeted obsolete
experiment grids and environment-specific paths.

Use the explicit Python module entry points documented in the root README.
