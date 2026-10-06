# Streaming Quantile

Approximate percentiles over a stream in constant memory.

```python
from streaming_quantile import QuantileEstimator

est = QuantileEstimator(0.5)  # median
for value in [1, 2, 3, 4, 5]:
    est.observe(value)
print(est.estimate())
```

## Why this library exists

Computing exact percentiles requires storing every value seen so far, which is
impossible for unbounded streams or memory-constrained environments.  The
estimator here uses the P² algorithm (Jain and Chlamtac, 1985), which keeps only
five marker values and updates them in O(1) time per observation.  The trade-off
is accuracy: the result is an approximation, not the exact quantile.  For smooth
or slowly changing distributions the error is typically a few percent, but
sudden shifts in the data can temporarily increase the error.

## Awkward edge

The estimator returns `float('nan')` until it has seen at least five
observations.  This is a deliberate choice: with fewer points the quantile is
not well-defined, and the internal markers are not yet initialized.  Calling
`estimate()` on a fresh estimator or after only a few observations will yield
`nan`, so check for that if your stream may be very short.

## Exports

The package exports one class:

- `QuantileEstimator(q)` — create an estimator for probability `q` in `(0, 1)`.
  Methods:
  - `observe(value)` — add a single numeric observation.
  - `observe_many(iterable)` — add all values from an iterable.
  - `estimate()` — return the current quantile estimate (or `nan`).

Only the standard library is used.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

