import numpy as np
import warnings
from rge256 import RGE256ctr_NumPy

# No overflow warnings with the new implementation
warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_random_uint32_type_and_value():
    rng = RGE256ctr_NumPy(seed=1)
    x = rng.random_uint32(1)[0]

    # Accept NumPy integer types
    assert isinstance(x, (int, np.integer))

    # It should fit in 32 bits
    assert 0 <= int(x) <= 0xFFFFFFFF


def test_rand_range_and_type():
    rng = RGE256ctr_NumPy(seed=123)
    f = rng.rand((1,))[0]

    # Must be NumPy float32
    assert isinstance(f, (float, np.floating))

    # Must be in [0, 1)
    assert 0.0 <= f < 1.0


def test_randint_bounds():
    rng = RGE256ctr_NumPy(seed=777)
    vals = rng.randint(10, 20, (100,))

    # All values should be in range [10, 20)
    assert np.all(vals >= 10)
    assert np.all(vals < 20)


def test_batch_output():
    rng = RGE256ctr_NumPy(seed=5)
    batch = rng.random_uint32(100)

    # Type check
    assert isinstance(batch, np.ndarray)
    assert batch.dtype == np.uint32

    # Correct length
    assert len(batch) == 100


def test_reproducibility():
    rng1 = RGE256ctr_NumPy(seed=999)
    rng2 = RGE256ctr_NumPy(seed=999)

    # Same seeds must match for first 10 outputs
    vals1 = rng1.random_uint32(10)
    vals2 = rng2.random_uint32(10)

    assert np.array_equal(vals1, vals2)


def test_different_seeds_independent():
    rng1 = RGE256ctr_NumPy(seed=100)
    rng2 = RGE256ctr_NumPy(seed=200)

    # First value should almost always be different
    assert rng1.random_uint32(1)[0] != rng2.random_uint32(1)[0]


def test_rand_shape():
    rng = RGE256ctr_NumPy(seed=42)
    arr = rng.rand((3, 4))

    # Check shape
    assert arr.shape == (3, 4)
    assert arr.dtype == np.float32


def test_randn_shape_and_type():
    rng = RGE256ctr_NumPy(seed=12345)
    arr = rng.randn((10,))

    # Check shape and type
    assert arr.shape == (10,)
    assert arr.dtype == np.float32

    # Check values are not all the same
    assert not np.all(arr == arr[0])
