from rge256 import RGE256
import numpy as np

def test_basic_usage():
    rng = RGE256(seed=1)
    x = rng.next32()
    assert isinstance(x, (int, np.integer))
