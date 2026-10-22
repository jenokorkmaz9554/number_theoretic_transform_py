import unittest

from number_theoretic_transform_py.core import (
    ntt,
    intt,
    ntt_convolve,
    find_primitive_root,
    is_primitive_root,
    is_prime,
)


class TestPrimesAndRoots(unittest.TestCase):

    def test_is_prime_small(self):
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(998244353))
        self.assertFalse(is_prime(1))
        self.assertFalse(is_prime(0))
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(9))

    def test_find_primitive_root(self):
        # 3 is a known primitive root of 998244353
        self.assertEqual(find_primitive_root(998244353), 3)

    def test_find_primitive_root_small(self):
        self.assertEqual(find_primitive_root(5), 2)
        self.assertEqual(find_primitive_root(7), 3)
        self.assertEqual(find_primitive_root(97), 5)

    def test_find_primitive_root_not_prime(self):
        with self.assertRaises(ValueError):
            find_primitive_root(4)

    def test_is_primitive_root(self):
        self.assertTrue(is_primitive_root(3, 998244353))
        self.assertFalse(is_primitive_root(1, 998244353))
        self.assertFalse(is_primitive_root(0, 998244353))
        # 9 * k + 1 property: 9 is not a primitive root of 998244353
        self.assertFalse(is_primitive_root(9, 998244353))

    def test_is_primitive_root_p2(self):
        self.assertTrue(is_primitive_root(1, 2))
        self.assertTrue(is_primitive_root(3, 2))
        self.assertFalse(is_primitive_root(0, 2))


class TestNTT(unittest.TestCase):

    PRIME = 998244353
    ROOT = 3

    def test_ntt_intt_roundtrip(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8]
        fa = ntt(a, self.PRIME, self.ROOT)
        recovered = intt(fa, self.PRIME, self.ROOT)
        self.assertEqual(recovered, [x % self.PRIME for x in a])

    def test_ntt_intt_roundtrip_large_values(self):
        a = [0, self.PRIME - 1, 1, self.PRIME]
        fa = ntt(a, self.PRIME, self.ROOT)
        recovered = intt(fa, self.PRIME, self.ROOT)
        expected = [x % self.PRIME for x in a]
        self.assertEqual(recovered, expected)

    def test_ntt_length_one(self):
        a = [42]
        fa = ntt(a, self.PRIME, self.ROOT)
        self.assertEqual(fa, [42])
        recovered = intt(fa, self.PRIME, self.ROOT)
        self.assertEqual(recovered, [42])

    def test_ntt_empty(self):
        self.assertEqual(ntt([], self.PRIME, self.ROOT), [])
        self.assertEqual(intt([], self.PRIME, self.ROOT), [])

    def test_ntt_non_power_of_two(self):
        with self.assertRaises(ValueError):
            ntt([1, 2, 3], self.PRIME, self.ROOT)

    def test_ntt_does_not_mutate_input(self):
        a = [1, 2, 3, 4]
        original = list(a)
        ntt(a, self.PRIME, self.ROOT)
        self.assertEqual(a, original)

    def test_ntt_negative_inputs(self):
        a = [-1, -2, 3, -4]
        fa = ntt(a, self.PRIME, self.ROOT)
        recovered = intt(fa, self.PRIME, self.ROOT)
        expected = [x % self.PRIME for x in a]
        self.assertEqual(recovered, expected)

    def test_ntt_known_value(self):
        # Verify NTT against direct DFT definition for n=4
        a = [1, 2, 3, 4]
        p = self.PRIME
        g = self.ROOT
        n = 4
        # primitive 4th root = g^((p-1)/4)
        w4 = pow(g, (p - 1) // n, p)
        expected = [
            sum(a[k] * pow(w4, j * k, p) for k in range(n)) % p
            for j in range(n)
        ]
        result = ntt(a, p, g)
        self.assertEqual(result, expected)


class TestConvolve(unittest.TestCase):

    PRIME = 998244353
    ROOT = 3

    def _brute_convolve(self, a, b, n):
        p = self.PRIME
        return [
            sum(a[i] * b[(k - i) % n] for i in range(n)) % p
            for k in range(n)
        ]

    def test_convolve_basic(self):
        a = [1, 2, 3, 4]
        b = [5, 6, 7, 8]
        result = ntt_convolve(a, b, self.PRIME, self.ROOT)
        expected = self._brute_convolve(a, b, 4)
        self.assertEqual(result, expected)

    def test_convolve_matches_brute_force(self):
        cases = [
            ([1, 1], [1, 1]),
            ([1, 2, 3, 4], [0, 0, 0, 1]),
            ([1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]),
            ([100, 200, 300, 400], [1, 0, 0, 0]),
        ]
        for a, b in cases:
            n = len(a)
            result = ntt_convolve(a, b, self.PRIME, self.ROOT)
            expected = self._brute_convolve(a, b, n)
            self.assertEqual(result, expected, f"Failed for a={a}, b={b}")

    def test_convolve_exact_integer(self):
        # Inputs small enough that sums stay well below prime.
        # Cyclic convolution wraps indices modulo n.
        a = [1, 2, 3, 4]
        b = [1, 2, 3, 4]
        result = ntt_convolve(a, b, self.PRIME, self.ROOT)
        expected = self._brute_convolve(a, b, 4)
        self.assertEqual(result, expected)

    def test_convolve_zero_operand(self):
        a = [0, 0, 0, 0]
        b = [1, 2, 3, 4]
        result = ntt_convolve(a, b, self.PRIME, self.ROOT)
        self.assertEqual(result, [0, 0, 0, 0])

    def test_convolve_both_zero(self):
        a = [0, 0]
        b = [0, 0]
        result = ntt_convolve(a, b, self.PRIME, self.ROOT)
        self.assertEqual(result, [0, 0])

    def test_convolve_empty(self):
        self.assertEqual(ntt_convolve([], [], self.PRIME, self.ROOT), [])

    def test_convolve_length_mismatch(self):
        with self.assertRaises(ValueError):
            ntt_convolve([1, 2], [1, 2, 3], self.PRIME, self.ROOT)

    def test_convolve_non_power_of_two(self):
        with self.assertRaises(ValueError):
            ntt_convolve([1, 2, 3], [1, 2, 3], self.PRIME, self.ROOT)

    def test_convolve_negative(self):
        a = [-1, 2, -3, 4]
        b = [1, -2, 3, -4]
        result = ntt_convolve(a, b, self.PRIME, self.ROOT)
        expected = self._brute_convolve(a, b, 4)
        self.assertEqual(result, expected)

    def test_convolve_modular_reduction(self):
        # Values at the edge of the modulus; convolution result must wrap
        p = self.PRIME
        a = [p - 1, p - 1, p - 1, p - 1]
        b = [p - 1, p - 1, p - 1, p - 1]
        result = ntt_convolve(a, b, p, self.ROOT)
        expected = self._brute_convolve(a, b, 4)
        self.assertEqual(result, expected)

    def test_convolve_does_not_mutate_inputs(self):
        a = [1, 2, 3, 4]
        b = [5, 6, 7, 8]
        a_orig = list(a)
        b_orig = list(b)
        ntt_convolve(a, b, self.PRIME, self.ROOT)
        self.assertEqual(a, a_orig)
        self.assertEqual(b, b_orig)


if __name__ == "__main__":
    unittest.main()
