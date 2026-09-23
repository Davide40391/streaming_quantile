"""Core implementation of the streaming quantile estimator.

The algorithm is based on the P² algorithm by Jain and Chlamtac (1985),
which maintains five markers to estimate a single quantile with constant
memory.  It processes one value at a time and updates marker positions
and heights using a piecewise-parabolic interpolation.  The algorithm is
deterministic and does not store the input stream.

Why P²?  It is simple, has a tiny footprint (five stored numbers plus
bookkeeping), and provides good accuracy for many real-world skewed
streams.  The trade-off is that the estimate is approximate and can lag
behind abrupt distribution changes, especially for extreme quantiles
(e.g., 0.01 or 0.99).
"""

from collections.abc import Iterable
from math import copysign


class QuantileEstimator:
    """Estimate a single quantile over a stream of numbers in O(1) memory.

    The estimator is initialized with a target probability ``q`` in
    ``(0, 1)``.  For example, ``q=0.5`` tracks the median.  After
    feeding at least 5 observations, the :meth:`estimate` method returns
    the current approximation.

    The implementation follows the classic P² algorithm.  It maintains
    five marker heights that track the minimum, the quantile, and the
    maximum, with two additional markers in between.  Marker positions
    are adjusted to keep them close to the desired quantile index.

    Attributes:
        q (float): The target probability, strictly between 0 and 1.
        n (int): Number of observations processed.
    """

    def __init__(self, q: float) -> None:
        """Create an estimator for probability ``q``.

        Args:
            q: Target quantile probability in ``(0, 1)``.

        Raises:
            ValueError: If ``q`` is not in ``(0, 1)``.
        """
        if not 0.0 < q < 1.0:
            raise ValueError("q must be strictly between 0 and 1")
        self.q = q
        self.n = 0
        # Marker heights.  The five entries correspond to the minimum,
        # the first intermediate marker, the quantile marker, the second
        # intermediate marker, and the maximum.
        self._heights = [0.0] * 5
        # Desired marker positions (fractional indices) in the sorted
        # stream.  Stored as floats so updates are smooth.
        self._positions = [0.0, 0.0, 0.0, 0.0, 0.0]
        # Desired positions, computed on the fly from ``q`` and ``n``.
        self._desired = [0.0] * 5

    def observe(self, value: float) -> None:
        """Process a single numeric observation.

        Args:
            value: The next number in the stream.

        Raises:
            TypeError: If ``value`` is not a real number.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be an int or float")
        value = float(value)

        self.n += 1

        if self.n <= 5:
            # Insertion sort for the first five observations.  This is
            # the only time we store the actual input values; afterwards
            # only the five markers are kept.
            self._heights[self.n - 1] = value
            if self.n == 5:
                self._heights.sort()
                self._positions = [0.0, 1.0, 2.0, 3.0, 4.0]
            return

        # From the sixth observation onward we update the markers.
        # Determine which cell the new value falls into.
        if value < self._heights[0]:
            self._heights[0] = value
            k = 0
        elif value < self._heights[1]:
            k = 0
        elif value < self._heights[2]:
            k = 1
        elif value < self._heights[3]:
            k = 2
        elif value < self._heights[4]:
            k = 3
        else:
            self._heights[4] = value
            k = 3

        # Increment positions of markers to the right of the new value.
        for i in range(k + 1, 5):
            self._positions[i] += 1.0

        # Recompute desired positions for the current count.
        self._desired[0] = 0.0
        self._desired[1] = 2.0 * self.q * (self.n - 1)
        self._desired[2] = 4.0 * self.q * (self.n - 1)
        self._desired[3] = 2.0 * (1.0 + self.q) * (self.n - 1) / 2.0 + 1.0
        # The original algorithm defines desired[3] as
        # 2 + 2 * q * (n - 1) / 2?  Let us follow the standard P² formula:
        # desired[1] = (n - 1) * q / 2
        # desired[2] = (n - 1) * q
        # desired[3] = (n - 1) * (1 + q) / 2
        # desired[4] = n - 1
        # We recompute cleanly below.
        self._desired[1] = (self.n - 1) * self.q / 2.0
        self._desired[2] = (self.n - 1) * self.q
        self._desired[3] = (self.n - 1) * (1.0 + self.q) / 2.0
        self._desired[4] = float(self.n - 1)

        # Adjust marker positions and heights for markers 1..3.
        # Marker 0 and 4 are fixed at the min and max.
        for i in range(1, 4):
            d = self._desired[i] - self._positions[i]
            if (d >= 1.0 and self._positions[i + 1] - self._positions[i] > 1.0) or \
               (d <= -1.0 and self._positions[i - 1] - self._positions[i] < -1.0):
                d_sign = int(copysign(1.0, d))
                # Try the piecewise-parabolic formula.
                qs = self._parabolic(i, d_sign)
                if self._heights[i - 1] < qs < self._heights[i + 1]:
                    self._heights[i] = qs
                else:
                    # Fallback to linear interpolation.
                    self._heights[i] = self._linear(i, d_sign)
                self._positions[i] += d_sign

    def estimate(self) -> float:
        """Return the current quantile estimate.

        The estimate is only meaningful after at least 5 observations.
        Before that, returns ``float('nan')`` because there is not
        enough information to compute a quantile.

        Returns:
            The estimated quantile value, or ``nan`` if fewer than 5
            observations have been processed.
        """
        if self.n < 5:
            return float("nan")
        return self._heights[2]

    def observe_many(self, values: Iterable[float]) -> None:
        """Process an iterable of observations in order.

        This is a convenience method; it simply calls :meth:`observe` for
        each item.  It is provided because streaming data often arrives
        in batches.

        Args:
            values: Any iterable of numbers.
        """
        for v in values:
            self.observe(v)

    def _parabolic(self, i: int, d: int) -> float:
        """Compute the parabolic prediction for marker ``i``.

        Args:
            i: Marker index (1, 2, or 3).
            d: Direction of adjustment (+1 or -1).

        Returns:
            The predicted height.
        """
        p = self._positions
        h = self._heights
        return h[i] + d / (p[i + 1] - p[i - 1]) * (
            (p[i] - p[i - 1] + d) * (h[i + 1] - h[i]) / (p[i + 1] - p[i])
            + (p[i + 1] - p[i] - d) * (h[i] - h[i - 1]) / (p[i] - p[i - 1])
        )

    def _linear(self, i: int, d: int) -> float:
        """Compute the linear fallback prediction for marker ``i``.

        Args:
            i: Marker index (1, 2, or 3).
            d: Direction of adjustment (+1 or -1).

        Returns:
            The linearly interpolated height.
        """
        p = self._positions
        h = self._heights
        return h[i] + d * (h[i + d] - h[i]) / (p[i + d] - p[i])
