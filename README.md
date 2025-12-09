

**Numpyrge256**

A pure NumPy implementation of **RGE-256**, a 256-bit ARX-based pseudorandom number generator featuring counter-mode operation, SHA-512 initialization, and structured entropy design.

This package provides a lightweight, high-performance core suitable for research, simulations, Monte Carlo methods, and general-purpose randomness in scientific computing.

**Author:** Steven Reid
**ORCID:** 0009-0003-9132-3410
**Paper:** *RGE-256: A New ARX-Based Pseudorandom Number Generator With Structured Entropy and Empirical Validation* (Nov 2025)
**Zenodo:** [https://zenodo.org/records/17713219](https://zenodo.org/records/17713219)
**PyTorch version:** [https://github.com/RRG314/torchrge256](https://github.com/RRG314/torchrge256)
**Demo:** [https://github.com/RRG314/RGE-256-app](https://github.com/RRG314/RGE-256-app)

---

## Key Features

* **Counter-Mode Operation**: 64-bit counter guarantees 2^64 unique blocks
* **Full 256-bit Output**: 8×32-bit words per block
* **Pure Python ARX Core**: No NumPy overflow warnings
* **SHA-512 Initialization**: Cryptographically-strong seed expansion
* **NumPy Array Outputs**: Direct tensor generation for scientific computing
* **Domain Separation**: Independent streams via domain parameter
* **Reproducible Sequences**: Deterministic outputs for research and simulation
* **Modern API**: `rand()`, `randint()`, `randn()` for NumPy-style workflows

This implementation prioritizes simplicity, portability, and reliability for scientific use cases.

---

## Installation

PyPI:

```bash
pip install numpyrge256
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/RRG314/numpyrge256
```

---

## Quick Start

```python
from rge256 import RGE256ctr_NumPy

# Create generator with seed
rng = RGE256ctr_NumPy(seed=12345)

# Generate random floats in [0, 1) - shape: (3, 4)
x = rng.rand((3, 4))

# Generate random integers in [0, 100) - shape: (10,)
y = rng.randint(0, 100, (10,))

# Generate normal distribution - shape: (1000,)
z = rng.randn((1000,))

# Generate raw uint32 values
raw = rng.random_uint32(100)
```

---

## API Summary

### `RGE256ctr_NumPy(seed=42, rounds=6, domain="rge256ctr")`

Creates a new RGE-256 counter-mode generator.

**Parameters:**
* `seed` (int): Seed value for initialization (default: 42)
* `rounds` (int): Number of ARX rounds per block (default: 6)
* `domain` (str): Domain separator for independent streams (default: "rge256ctr")

**Core Methods:**

* **`random_uint32(n)`** – Returns NumPy array of `n` random uint32 values
  ```python
  arr = rng.random_uint32(100)  # shape: (100,), dtype: uint32
  ```

* **`rand(shape)`** – Returns NumPy array of floats in [0, 1)
  ```python
  arr = rng.rand((5, 10))  # shape: (5, 10), dtype: float32
  ```

* **`randint(low, high, shape)`** – Returns NumPy array of integers in [low, high)
  ```python
  arr = rng.randint(0, 100, (20,))  # shape: (20,), dtype: int64
  ```

* **`randn(shape)`** – Returns NumPy array with standard normal distribution
  ```python
  arr = rng.randn((1000,))  # shape: (1000,), dtype: float32, mean≈0, std≈1
  ```

All outputs are deterministic given `(seed, domain, rounds)`.

---

## Version 1.1.0 Updates

**New in v1.1.0:**

* **Counter-Mode Architecture**: Stateless block generation with 64-bit counter
* **SHA-512 Initialization**: Replaced simple hash with cryptographic-grade seed expansion
* **Pure Python ARX**: Eliminates NumPy overflow warnings
* **NumPy-Style API**: New `rand()`, `randint()`, `randn()` methods
* **Enhanced Security**: Increased default rounds from 3 to 6
* **Full 256-bit Blocks**: All 8×32-bit words available per block

**Migration from v1.0.0:**

```python
# Old API (v1.0.0)
from rge256 import RGE256
rng = RGE256(seed=42, rounds=3)
x = rng.next32()            # Single value
f = rng.nextFloat()         # Single float
batch = rng.next32_batch(100)

# New API (v1.1.0)
from rge256 import RGE256ctr_NumPy
rng = RGE256ctr_NumPy(seed=42, rounds=6)
x = rng.random_uint32(1)[0]  # Array interface
f = rng.rand((1,))[0]        # Array interface
batch = rng.random_uint32(100)  # Direct replacement
```

---

## Statistical Properties

The counter-mode implementation maintains strong statistical properties:

* **Period**: 2^64 blocks × 256 bits = 2^72 bits (18 exabytes)
* **Entropy**: ≈7.999–8.000 bits/byte
* **Bit Distribution**: Uniform (~50% ones per bit position)
* **Correlation**: Low serial and lag-1 correlation
* **Reproducibility**: Identical outputs across platforms

Empirical testing includes Dieharder, bit balance analysis, and chi-square evaluation.

---

## Example: Monte Carlo Simulation

```python
from rge256 import RGE256ctr_NumPy
import numpy as np

# Estimate π using Monte Carlo
rng = RGE256ctr_NumPy(seed=2025)
n = 1_000_000

# Generate random points in unit square
x = rng.rand((n,))
y = rng.rand((n,))

# Count points inside unit circle
inside = (x**2 + y**2) <= 1.0
pi_estimate = 4 * np.sum(inside) / n

print(f"π estimate: {pi_estimate:.6f}")
```

---

## Disclaimer

RGE-256 is **not** designed as or intended to serve as a cryptographic random number generator.
It has not been formally analyzed for cryptographic security.

Use only for research, simulation, and non-security-critical applications.

---

## Citation

Please cite the corresponding preprint:

```bibtex
@misc{reid2025rge256,
  author       = {Reid, Steven},
  title        = {RGE-256: A New ARX-Based Pseudorandom Number Generator
                  With Structured Entropy and Empirical Validation},
  year         = {2025},
  howpublished = {\url{https://zenodo.org/records/17713219}},
  note         = {ORCID: 0009-0003-9132-3410}
}
```

---

## Related Repositories

* **PyTorch implementation:**
  [https://github.com/RRG314/torchrge256](https://github.com/RRG314/torchrge256)

* **Web-based demonstration:**
  [https://github.com/RRG314/RGE-256-app](https://github.com/RRG314/RGE-256-app)

---

## License

MIT license.

