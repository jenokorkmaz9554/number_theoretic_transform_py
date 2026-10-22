"""Number Theoretic Transform over a prime field.

Provides exact integer convolution via NTT.  Results are correct modulo the
chosen prime; for plain integer convolution, the sum of products must be
below the prime so the residue equals the true value.
"""

from __future__ import annotations


def _lg(n: int) -> int:
    """Ceiling log2 of n, n >= 1."""
    return n.bit_length() - 1 if (n & (n - 1)) == 0 else n.bit_length()


def _pow_mod(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation.  Python's builtin pow() does exactly this and is
    implemented in C, so delegating is both clearer and faster than a Python
    square-and-multiply loop."""
    return pow(base, exp, mod)


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 3.3e24 using the first 13 primes.

    This covers every modulus this library is likely to see in practice.
"""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def is_primitive_root(g: int, p: int) -> bool:
    """True iff g is a primitive root modulo prime p.

    Checks g^((p-1)/q) != 1 for every prime q dividing p-1.  The first 12
    primes suffice to factor p-1 for any p this library targets (p < 2^60).
    """
    if p < 2:
        return False
    if p == 2:
        return g % p == 1
    if g % p == 0:
        return False
    phi = p - 1
    factors = set()
    m = phi
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if m == 1:
            break
        if m % q == 0:
            factors.add(q)
            while m % q == 0:
                m //= q
    if m != 1:
        raise ValueError(
            f"p-1 = {phi} has a prime factor > 37; cannot test primitive root"
        )
    g %= p
    for q in factors:
        if pow(g, phi // q, p) == 1:
            return False
    return True


def find_primitive_root(p: int) -> int:
    """Return the smallest primitive root modulo prime p.

    Raises ValueError if p is not prime.  Assumes p-1's prime factors are all
    <= 37 (true for the moduli this library is designed for)."""
    if not is_prime(p):
        raise ValueError(f"{p} is not prime")
    if p == 2:
        return 1
    for g in range(2, p):
        if is_primitive_root(g, p):
            return g
    raise RuntimeError(f"unreachable: no primitive root found for prime {p}")


def _bit_reverse(data: list, bits: int) -> None:
    """In-place iterative bit-reversal permutation.

    A recursive NTT is clearer but risks stack overflow on large transforms
    and is slower in CPython.  The iterative form with explicit bit reversal
    is the standard production approach."""
    n = len(data)
    for i in range(n):
        j = 0
        x = i
        for _ in range(bits):
            j = (j << 1) | (x & 1)
            x >>= 1
        if i < j:
            data[i], data[j] = data[j], data[i]


def ntt(a: list, prime: int, root: int, invert: bool = False) -> list:
    """Forward or inverse Number Theoretic Transform of a modulo prime.

    Length must be a power of two dividing p-1, and root must be a primitive
    len(a)-th root of unity modulo prime.  Returns a new list; the input is
    not modified.

    Setting invert=True computes the inverse transform, including division
    by n (as multiplication by n^{-1}).
    """
    n = len(a)
    if n == 0:
        return []
    if n & (n - 1):
        raise ValueError(f"length {n} is not a power of two")
    a_mod = [x % prime for x in a]

    if n == 1:
        return list(a_mod)

    bits = _lg(n)
    _bit_reverse(a_mod, bits)

    w = root % prime
    if invert:
        w = pow(w, prime - 2, prime)

    length = 2
    while length <= n:
        half = length // 2
        # Primitive length-th root of unity:  w^((p-1)/length) mod p
        step = (prime - 1) // length
        wn = pow(w, step, prime)
        for i in range(0, n, length):
            wi = 1
            for j in range(half):
                u = a_mod[i + j]
                v = a_mod[i + j + half] * wi % prime
                a_mod[i + j] = (u + v) % prime
                a_mod[i + j + half] = (u - v) % prime
                wi = wi * wn % prime
        length *= 2

    if invert:
        n_inv = pow(n, prime - 2, prime)
        a_mod = [x * n_inv % prime for x in a_mod]

    return a_mod


def intt(a: list, prime: int, root: int) -> list:
    """Inverse NTT.  Equivalent to ntt(a, prime, root, invert=True) but named
    for readability at call sites."""
    return ntt(a, prime, root, invert=True)


def ntt_convolve(a: list, b: list, prime: int, root: int) -> list:
    """Cyclic convolution of a and b modulo prime via NTT.

    Both inputs must have equal, power-of-two length n.  Returns a list of
    length n.  Result[k] = sum(a[i]*b[(k-i) mod n] for i in range(n)) mod prime.

    When at least one operand is zero, the implementation short-circuits to
    an all-zero result rather than performing the transform — this is both
    faster and avoids any concern about transforms of empty-style data.
    """
    n = len(a)
    if n != len(b):
        raise ValueError(f"length mismatch: {n} vs {len(b)}")
    if n & (n - 1):
        raise ValueError(f"length {n} is not a power of two")
    if n == 0:
        return []

    result_len = n

    if not any(x % prime for x in a) or not any(x % prime for x in b):
        return [0] * result_len

    fa = ntt(a, prime, root)
    fb = ntt(b, prime, root)
    fc = [(fa[i] * fb[i]) % prime for i in range(n)]
    c = intt(fc, prime, root)
    return c[:result_len]
