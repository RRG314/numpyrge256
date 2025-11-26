import numpy as np
import warnings
from rge256 import RGE256

# Silence expected NumPy overflow warnings from ARX math
warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_next32_type_and_value():
    rng = RGE256(seed=1)
    x = rng.next32()

    # Accept NumPy integer types and Python int
    assert isinstance(x, (int, np.integer))

    # It should fit in 32 bits
    assert 0 <= int(x) <= 0xFFFFFFFF


def test_nextFloat_range_and_type():
    rng = RGE256(seed=123)
    f = rng.nextFloat()

    # Must be a Python float
    assert isinstance(f, float)

    # Must be in [0, 1)
    assert 0.0 <= f < 1.0


def test_nextRange_bounds():
    rng = RGE256(seed=777)
    for _ in range(100):
        v = rng.nextRange(10, 20)
        assert 10 <= v <= 20


def test_batch_output():
    rng = RGE256(seed=5)
    batch = rng.next32_batch(100)

    # Type check
    assert isinstance(batch, np.ndarray)
    assert batch.dtype == np.uint32

    # Correct length
    assert len(batch) == 100


def test_reproducibility():
    rng1 = RGE256(seed=999)
    rng2 = RGE256(seed=999)

    # Same seeds must match for first 10 outputs
    vals1 = [rng1.next32() for _ in range(10)]
    vals2 = [rng2.next32() for _ in range(10)]

    assert vals1 == vals2


def test_different_seeds_independent():
    rng1 = RGE256(seed=100)
    rng2 = RGE256(seed=200)

    # First value should almost always be different
    assert rng1.next32() != rng2.next32()
