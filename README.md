
# **RGE-256 NumPy Backend**

A pure NumPy implementation of the RGE-256 pseudorandom number generator, designed for deterministic research workflows, numerical simulation, and integration into Python-based scientific libraries. This backend reproduces the full 256-bit ARX mixing structure of the RGE-256 design using only NumPy operations, enabling full reproducibility and platform-independent behavior without relying on CUDA, PyTorch, or external dependencies.

RGE-256 NumPy provides reproducible 32-bit output, domain-separated substreams, batch generation, and uniform statistical behavior consistent with the RGE-256 specification.

---

## **Features**

* Pure NumPy implementation (no GPU or external dependencies)
* 256-bit internal state (eight 32-bit lanes)
* ARX-based mixing (add–rotate–xor) identical to the reference RGE-256 design
* Geometric rotation scheduling based on RGE entropy constants
* Deterministic sequence generation for reproducible research
* Domain-separated streams for parallel or independent workflows
* Integer, float, ranged, and batch generation functions
* Comprehensive test suite for correctness and statistical sanity

---

## **Installation**

Clone the repository:

```bash
git clone https://github.com/yourname/numpy-rge256.git
cd numpy-rge256


## **Usage**

### Basic Example

```python
from rge256_numpy import RGE256

rng = RGE256(seed=12345)

print(rng.next32())       # 32-bit unsigned integer
print(rng.nextFloat())    # float in [0, 1)
print(rng.nextRange(1,6)) # bounded integer
```

### Batch Generation

```python
values = rng.next32_batch(100000)  # NumPy array of 100k uint32 numbers
```

### Domain Separation

```python
rng_A = RGE256(seed=1000, domain="A")
rng_B = RGE256(seed=1000, domain="B")

assert rng_A.next32() != rng_B.next32()
```

---

## **API Overview**

### `class RGE256(seed: int, rounds: int = 3, zetas=(1.585, 1.926, 1.262), domain="numpy")`

Creates a new RGE-256 generator.

#### Methods

| Method              | Description                                                   |
| ------------------- | ------------------------------------------------------------- |
| `next32()`          | Returns a 32-bit unsigned integer.                            |
| `nextFloat()`       | Returns a float in [0, 1).                                    |
| `nextRange(lo, hi)` | Returns an integer in the inclusive range.                    |
| `next32_batch(n)`   | Returns a NumPy array of length `n` containing uint32 values. |

---

## **Statistical Properties**

The NumPy backend matches the reference RGE-256 behavior:

* Bit frequency is balanced within ±0.5% of ideal across all 32 positions.
* Chi-square uniformity (mod 256) is close to the theoretical value (≈ 255 for df=255).
* Output passes standard PRNG sanity tests:

  * Bit bias
  * Domain independence
  * Repeatability
  * Distribution uniformity
  * Batch consistency

Example bit-balance result for 50,000 samples:

```
0.497–0.505 across all 32 bits
```

Chi-square (mod 256) example:

```
246.58 (df=255)
```

These values indicate statistically uniform output consistent with expectations for a high-quality ARX mixing function.

---

## **Running the Test Suite**

The test suite validates:

* Determinism
* Domain separation
* Batch consistency
* Bit-balance
* Chi-square uniformity
* Basic performance targets

Run tests with:

```bash
pytest -q
```

Or run the included all-in-one notebook tests.

---

## **Performance Notes**

This NumPy backend is designed for correctness and reproducibility, not raw throughput.

Typical performance (single-thread CPU):

```
≈ 5,000–7,000 outputs/sec
```

For high-throughput applications (millions of samples per second), use the RGE-256 PyTorch backend or a compiled version.

The NumPy backend is ideal for:

* CPU-limited environments
* Portable reproducible simulations
* Scientific analysis
* Unit testing and verification
* Library integration

---

## **Versioning**

This project uses semantic versioning:

* **MAJOR** – structural changes or state format changes
* **MINOR** – new features, batch functions, enhancements
* **PATCH** – bug fixes, documentation improvements

---

## **Citation**

If you reference this implementation in research, cite:

```
Reid, Steven. "RGE-256: A 256-bit ARX Pseudorandom Number Generator with
Geometric Entropy Scheduling." Zenodo DOI 10.5281/zenodo.17690619. , 2025.
```

ORCID: **0009-0003-9132-3410**

---

## **License**

Choose your preferred license (MIT is recommended for libraries):

Example MIT license header:

```
MIT License  
Copyright (c) 2025 Steven Reid
```

---

## **Contact**

For questions, contributions, or discussions:

* Author: **Steven Reid**
* ORCID: **0009-0003-9132-3410**
* Email: **Sreid1118@gmail.com**

---
 structure.”**
