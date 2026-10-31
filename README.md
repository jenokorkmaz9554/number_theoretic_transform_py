# number_theoretic_transform_py

Exact integer convolution via the Number Theoretic Transform over a prime finite field.

```python
from number_theoretic_transform_py import ntt, intt, ntt_convolve, find_primitive_root

prime = 998244353
root = find_primitive_root(prime)

a = [1, 2, 3, 4]
b = [1, 2, 3, 4]
result = ntt_convolve(a, b, prime, root)
# result == [1, 4, 10, 20]
```

## Why this exists

Floating-point FFT introduces rounding error that accumulates unpredictably
for large integer convolutions.  NTT performs the same butterfly computation
in a prime field, so every operation is exact modular arithmetic — there is
no representation error and no accumulation drift.

The trade-off: results are correct modulo the chosen prime.  For plain
integer convolution, the prime must exceed the maximum possible output
coefficient.  `998244353` is provided as a common default (it is a
Proth prime with `p - 1 = 119 * 2^23`, supporting transforms up to length
2^23).

## What you need to know

- Input length must be a power of two.
- The prime `p` must satisfy `length | (p - 1)`, and `root` must be a
  primitive root modulo `p` (use `find_primitive_root`).
- For exact (non-modular) results, ensure `sum(|a[i]| * |b[i]|) < p` across
  each output coefficient.
- `is_primitive_root` factors `p - 1` using primes up to 37; it raises for
  primes whose `p - 1` has larger prime factors.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

