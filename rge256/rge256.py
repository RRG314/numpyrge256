import numpy as np
import hashlib
import math

MASK32 = (1 << 32) - 1


def rotl32(x: int, r: int) -> int:
    r &= 31
    x &= MASK32
    return ((x << r) & MASK32) | (x >> (32 - r))


class RGE256ctr_NumPy:
    """
    RGE-256 Counter-Mode — NumPy version
    ------------------------------------
    - Pure Python int ARX core (no overflow warnings)
    - 64-bit counter-mode (guaranteed 2^64 blocks)
    - 8×32-bit output per block (full 256-bit)
    - NumPy tensor outputs
    """

    K_BASE = [
        0x9E3779B9, 0x517CC1B7, 0xC2B2AE35, 0x165667B1,
        0x85EBCA77, 0x27D4EB2F, 0xDE5FB9D7, 0x94D049BB
    ]

    R = [13, 7, 11, 17, 19, 23, 29, 31]

    def __init__(self, seed=42, rounds=6, domain="rge256ctr"):
        self.seed = int(seed)
        self.rounds = int(rounds)
        self.domain = domain

        self.key = [0] * 8
        self.kmix = [0] * 8
        self.counter = 0

        self._init_from_seed(seed)

    # -------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------
    def _init_from_seed(self, seed):
        seed_bytes = (
            self.domain.encode("utf-8") +
            int(seed).to_bytes(8, "big", signed=True)
        )
        h = hashlib.sha512(seed_bytes).digest()

        words = []
        for i in range(8):
            w = int.from_bytes(h[i*4:(i+1)*4], "little") & MASK32
            words.append(w)

        self.key = words
        self.kmix = [(self.K_BASE[i] ^ words[i]) & MASK32 for i in range(8)]
        self.counter = 0

    # -------------------------------------------------------------
    # Block function F(key, counter) → 8×uint32 words
    # -------------------------------------------------------------
    def _block(self):
        s = self.key.copy()

        ctr = self.counter & ((1 << 64) - 1)
        ctr_lo = ctr & MASK32
        ctr_hi = (ctr >> 32) & MASK32

        s[0] = (s[0] + ctr_lo) & MASK32
        s[1] = (s[1] + ctr_hi) & MASK32

        for _ in range(self.rounds):

            # Quad ARX block
            for i in range(0, 8, 2):
                s[i] = (s[i] + self.kmix[i]) & MASK32
                s[i+1] ^= rotl32(s[i], self.R[i])

            # Cross coupling
            s[1] ^= rotl32(s[5], 13)
            s[3] ^= rotl32(s[7], 7)
            s[5] ^= rotl32(s[1], 11)
            s[7] ^= rotl32(s[3], 17)

            # Global fold
            s[0] ^= rotl32(s[4], 13)
            s[1] ^= rotl32(s[5], 19)
            s[2] ^= rotl32(s[6], 7)
            s[3] ^= rotl32(s[7], 23)

        # Increment counter
        self.counter = (self.counter + 1) & ((1 << 64) - 1)

        return s

    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------
    def random_uint32(self, n):
        out = []
        need = int(n)

        while need > 0:
            blk = self._block()
            take = min(8, need)
            out.extend(blk[:take])
            need -= take

        return np.array(out, dtype=np.uint32)

    def rand(self, shape):
        numel = int(np.prod(shape))
        vals = self.random_uint32(numel).astype(np.float64)
        return (vals / 2**32).reshape(shape).astype(np.float32)

    def randint(self, low, high, shape):
        numel = int(np.prod(shape))
        vals = self.random_uint32(numel)
        return (low + (vals % (high - low))).reshape(shape).astype(np.int64)

    def randn(self, shape):
        u1 = self.rand(shape)
        u2 = self.rand(shape)
        u1 = np.maximum(u1, 1e-12)
        return (
            np.sqrt(-2*np.log(u1)) * np.cos(2*np.pi*u2)
        ).astype(np.float32)
