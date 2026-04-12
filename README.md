# numpyrge256

[![GitHub Release](https://img.shields.io/github/v/release/RRG314/numpyrge256)](https://github.com/RRG314/numpyrge256/releases)
[![PyPI version](https://img.shields.io/pypi/v/numpyrge256)](https://pypi.org/project/numpyrge256/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](./LICENSE)

`numpyrge256` is the NumPy-native package in the RGE256 ecosystem. It provides a clean counter-mode generator for deterministic array generation in scientific Python workflows without requiring PyTorch or the full multi-variant suite.

## Current Release

**Current repository release:** `v1.1.1`

This release is a documentation and repo-surface refresh. The package API remains the same as the existing counter-mode NumPy implementation.

## When To Use This Repo

Use `numpyrge256` if you want:
- deterministic NumPy arrays for research workflows
- a lightweight package surface focused on one clear generator path
- reproducible Monte Carlo or simulation inputs in ordinary scientific Python environments

If you want the broader implementation family instead, use [rge256](https://github.com/RRG314/rge256).
If you want GPU- and tensor-oriented integration, use [torchrge256](https://github.com/RRG314/torchrge256).

## Installation

```bash
pip install numpyrge256
```

Or from GitHub:

```bash
pip install git+https://github.com/RRG314/numpyrge256
```

## Quick Start

```python
from rge256 import RGE256ctr_NumPy

rng = RGE256ctr_NumPy(seed=12345)

u = rng.rand((3, 4))
i = rng.randint(0, 100, (10,))
n = rng.randn((1000,))
raw = rng.random_uint32(16)
```

All outputs are deterministic given the same `(seed, rounds, domain)` configuration.

## What The Package Provides

- counter-mode block generation with 64-bit counter progression
- deterministic seed expansion and reproducible stream behavior
- direct NumPy-friendly sampling APIs
- domain separation support for independent streams
- a small package surface that is easy to install and reason about

## Validation

The repository includes a package-level test suite aimed at practical correctness and stable API behavior. This package should be read as a research and simulation tool rather than a cryptographic package.

## Repository Map

- `rge256/` - NumPy package implementation
- `tests/` - package validation checks
- `docs/releases/` - release notes for the GitHub repo surface

## Related Repositories

- [Recursive Geometric Entropy research program](https://github.com/RRG314/Recursive-Geometric-Entropy-research)
- [rge256](https://github.com/RRG314/rge256)
- [torchrge256](https://github.com/RRG314/torchrge256)
- [RGE-256-app](https://github.com/RRG314/RGE-256-app)
