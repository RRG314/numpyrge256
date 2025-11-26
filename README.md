

**Numpyrge256**

A pure NumPy implementation of **RGE-256**, a 256-bit ARX-based pseudorandom number generator featuring geometric rotation scheduling and structured entropy from Recursive Division Tree (RDT) analysis.

This package provides a lightweight, dependency-free core suitable for research, simulations, Monte Carlo methods, and general-purpose randomness in scientific computing.

**Author:** Steven Reid
**ORCID:** 0009-0003-9132-3410
**Paper:** *RGE-256: A New ARX-Based Pseudorandom Number Generator With Structured Entropy and Empirical Validation* (Nov 2025)
**Zenodo:** [https://zenodo.org/records/17713219](https://zenodo.org/records/17713219)
**PyTorch version:** [https://github.com/RRG314/torchrge256](https://github.com/RRG314/torchrge256)
**Demo:** [https://github.com/RRG314/RGE-256-app](https://github.com/RRG314/RGE-256-app)

---

## Key Features

* Pure NumPy implementation (no external dependencies)
* 256-bit internal state (8 × 32-bit words)
* Deterministic ARX update structure
* Rotation constants derived from structured geometric entropy
* Domain separation for independent streams
* Reproducible sequences for research and simulation
* Batch generation for large datasets

This implementation is focused on simplicity, portability, and reliability for scientific use cases.

---

## Installation

PyPI:

```
pip install rge256
```

Or install directly from GitHub:

```
pip install git+https://github.com/RRG314/numpyrge256
```

---

## Quick Start

```python
from rge256 import RGE256

rng = RGE256(seed=12345)

# Single 32-bit integer
x = rng.next32()

# Float in [0, 1)
f = rng.nextFloat()

# Integer in range [lo, hi]
v = rng.nextRange(1, 100)

# Batch of numbers
batch = rng.next32_batch(1000)
```

---

## API Summary

### `RGE256(seed, rounds=3, zetas=(1.585, 1.926, 1.262), domain="numpy")`

Creates a new RGE-256 generator.

Core methods:

* `next32()` – returns a 32-bit unsigned integer
* `nextFloat()` – returns a float in [0, 1)
* `nextRange(lo, hi)` – returns an integer in [lo, hi]
* `next32_batch(n)` – generates an array of n random uint32 values

All outputs are deterministic given `(seed, domain, rounds, zetas)`.

---

## Notes on Statistical Behavior

Empirical testing (Dieharder, bit balance analysis, chi-square evaluation) shows:

* Entropy ≈ 7.999–8.000 bits/byte
* Uniform bit distribution (~50% ones per bit position)
* Low serial and lag-1 correlation
* Stable behavior across NumPy and PyTorch implementations

The design inherits rotation structure from geometric entropy constants used in the corresponding RDT entropy framework.

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


