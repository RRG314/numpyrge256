from rge256 import RGE256

def test_basic_usage():
    rng = RGE256(seed=1)
    assert isinstance(rng.next32(), int)
