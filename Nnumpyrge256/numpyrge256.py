import numpy as np


def rotl32(x, r):
    r &= 31
    return ((x << r) | (x >> (32 - r))) & np.uint32(0xFFFFFFFF)


def simple_hash(s: str):
    h1 = np.uint32(0xDEADBEEF)
    h2 = np.uint32(0x41C6CE57)

    for ch in s:
        c = np.uint32(ord(ch))
        h1 = np.uint32((h1 ^ c) * np.uint32(2654435761))
        h2 = np.uint32((h2 ^ c) * np.uint32(1597334677))

    h1 = np.uint32((h1 ^ (h1 >> 16)) * np.uint32(2246822507)) ^ \
         np.uint32((h2 ^ (h2 >> 13)) * np.uint32(3266489909))

    h2 = np.uint32((h2 ^ (h2 >> 16)) * np.uint32(2246822507)) ^ \
         np.uint32((h1 ^ (h1 >> 13)) * np.uint32(3266489909))

    return h1, h2


class RGE256:
    """
    Pure NumPy implementation of the RGE-256 PRNG.
    Suitable for library packaging.
    """

    def __init__(self, seed: int, rounds=3,
                 zetas=(1.585, 1.926, 1.262),
                 domain="numpy"):

        self.rounds = int(rounds)

        h1, h2 = simple_hash(str(seed) + domain)

        state = np.zeros(16, dtype=np.uint32)
        x = np.uint32(seed)

        for i in range(16):
            x ^= np.uint32(x << 13)
            x ^= np.uint32(x >> 17)
            x ^= np.uint32(x << 5)
            x = np.uint32(x + h1 + h2 * np.uint32(i))
            state[i] = x

        self.s0, self.s1, self.s2, self.s3, \
        self.s4, self.s5, self.s6, self.s7 = state[:8]

        tri, meng, tet = zetas
        base = [
            tri, meng, tet,
            (tri + meng + tet) / 3,
            tri + 0.5, meng + 0.75, tet + 0.25,
            (tri * 1.25 + tet * 0.75) / 2
        ]

        rvals = []
        for i, z in enumerate(base):
            v = int(abs(z * 977 + (i * 7 + 13))) % 31
            rvals.append(1 if v == 0 else v)
        self.r = np.array(rvals, dtype=np.uint32)

        kbase = np.array([
            0x9E3779B9, 0x517CC1B7, 0xC2B2AE35, 0x165667B1,
            0x85EBCA77, 0x27D4EB2F, 0xDE5FB9D7, 0x94D049BB
        ], dtype=np.uint32)

        self.k = kbase ^ state[8:16]

        for _ in range(10):
            self._step()

    def _step(self):
        r = self.r
        k = self.k

        s0 = self.s0; s1 = self.s1; s2 = self.s2; s3 = self.s3
        s4 = self.s4; s5 = self.s5; s6 = self.s6; s7 = self.s7

        # Quad A
        s0 = np.uint32(s0 + s1 + k[0])
        s1 = rotl32(s1 ^ s0, r[0])
        s2 = np.uint32(s2 + s3 + k[1])
        s3 = rotl32(s3 ^ s2, r[1])

        tA0 = s0 ^ s2
        tA1 = s1 ^ s3
        s0 = np.uint32(s0 + rotl32(tA0, r[2]) + k[2])
        s2 = np.uint32(s2 + rotl32(tA1, r[3]) + k[3])
        s1 ^= rotl32(s0, (r[0] + r[2]) % 31 or 1)
        s3 ^= rotl32(s2, (r[1] + r[3]) % 31 or 1)

        # Quad B
        s4 = np.uint32(s4 + s5 + k[4])
        s5 = rotl32(s5 ^ s4, r[4])
        s6 = np.uint32(s6 + s7 + k[5])
        s7 = rotl32(s7 ^ s6, r[5])

        tB0 = s4 ^ s6
        tB1 = s5 ^ s7
        s4 = np.uint32(s4 + rotl32(tB0, r[6]) + k[6])
        s6 = np.uint32(s6 + rotl32(tB1, r[7]) + k[7])
        s5 ^= rotl32(s4, (r[4] + r[6]) % 31 or 1)
        s7 ^= rotl32(s6, (r[5] + r[7]) % 31 or 1)

        # Cross coupling
        s1 ^= rotl32(s5, 13)
        s3 ^= rotl32(s7, 7)
        s5 ^= rotl32(s1, 11)
        s7 ^= rotl32(s3, 17)

        # Global fold
        m0 = s0 ^ s4
        m1 = s1 ^ s5
        m2 = s2 ^ s6
        m3 = s3 ^ s7

        s0 = np.uint32(s0 + rotl32(m1, 3))
        s1 = np.uint32(s1 + rotl32(m2, 5))
        s2 = np.uint32(s2 + rotl32(m3, 7))
        s3 = np.uint32(s3 + rotl32(m0, 11))

        s4 ^= rotl32(s0, 19)
        s5 ^= rotl32(s1, 23)
        s6 ^= rotl32(s2, 29)
        s7 ^= rotl32(s3, 31)

        self.s0, self.s1, self.s2, self.s3 = s0, s1, s2, s3
        self.s4, self.s5, self.s6, self.s7 = s4, s5, s6, s7

    def next32(self) -> np.uint32:
        for _ in range(self.rounds):
            self._step()
        return np.uint32(self.s0 ^ rotl32(self.s4, 13))

    def nextFloat(self) -> float:
        return float(self.next32()) / 2**32

    def nextRange(self, lo: int, hi: int) -> int:
        return int(lo + (int(self.next32()) % (hi - lo + 1)))

    def next32_batch(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.uint32)
        for i in range(n):
            out[i] = self.next32()
        return out
