#!/usr/bin/env python3
"""Exact QDB 2019 one-round MF witness over Q(sqrt(5)).

This script verifies the closed-form primal/dual certificate

    p_MF^*(QDB 2019) = (5 + sqrt(5)) / 8

for the reduced one-round lifted-comb SDP used in the benchmark artifact.
It is intentionally self-contained and uses only exact arithmetic in
Q(sqrt(5)); no numerical SDP solver is called.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction

sys.set_int_max_str_digits(0)


@dataclass(frozen=True)
class Q5:
    """Element a + b sqrt(5), with a,b rational."""

    a: Fraction = Fraction(0)
    b: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", Fraction(self.a))
        object.__setattr__(self, "b", Fraction(self.b))

    def __add__(self, other: object) -> Q5:
        other = q5(other)
        return Q5(self.a + other.a, self.b + other.b)

    __radd__ = __add__

    def __neg__(self) -> Q5:
        return Q5(-self.a, -self.b)

    def __sub__(self, other: object) -> Q5:
        return self + (-q5(other))

    def __rsub__(self, other: object) -> Q5:
        return q5(other) - self

    def __mul__(self, other: object) -> Q5:
        other = q5(other)
        return Q5(
            self.a * other.a + 5 * self.b * other.b,
            self.a * other.b + self.b * other.a,
        )

    __rmul__ = __mul__

    def inv(self) -> Q5:
        den = self.a * self.a - 5 * self.b * self.b
        if den == 0:
            raise ZeroDivisionError("zero in Q(sqrt(5))")
        return Q5(self.a / den, -self.b / den)

    def __truediv__(self, other: object) -> Q5:
        return self * q5(other).inv()

    def __rtruediv__(self, other: object) -> Q5:
        return q5(other) * self.inv()

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0

    def sign(self) -> int:
        """Exact sign under the positive embedding sqrt(5)>0."""
        a, b = self.a, self.b
        if a == 0 and b == 0:
            return 0
        if b == 0:
            return 1 if a > 0 else -1
        if a >= 0 and b > 0:
            return 1
        if a <= 0 and b < 0:
            return -1

        # Opposite signs: compare |a| with |b| sqrt(5) by squaring.
        cmp = a * a - 5 * b * b
        if a > 0 and b < 0:
            return 1 if cmp > 0 else -1
        if a < 0 and b > 0:
            return 1 if cmp < 0 else -1
        raise AssertionError((a, b))

    def to_float(self) -> float:
        return float(self.a) + float(self.b) * math.sqrt(5.0)

    def complexity_bits(self) -> int:
        return max(
            abs(self.a.numerator).bit_length(),
            self.a.denominator.bit_length(),
            abs(self.b.numerator).bit_length(),
            self.b.denominator.bit_length(),
        )

    def __str__(self) -> str:
        if self.is_zero():
            return "0"
        if self.b == 0:
            return str(self.a)
        if self.a == 0:
            return f"{self.b}*sqrt5"
        sign = "+" if self.b > 0 else "-"
        return f"{self.a} {sign} {abs(self.b)}*sqrt5"


def q5(value: object) -> Q5:
    if isinstance(value, Q5):
        return value
    return Q5(Fraction(value), Fraction(0))


ZERO = Q5()
ONE = Q5(1)
SQRT5 = Q5(0, 1)


Matrix = list[list[Q5]]


def q(a: int | Fraction = 0, b: int | Fraction = 0) -> Q5:
    return Q5(Fraction(a), Fraction(b))


def zeros(n: int, m: int) -> Matrix:
    return [[ZERO for _ in range(m)] for _ in range(n)]


def eye(n: int) -> Matrix:
    out = zeros(n, n)
    for i in range(n):
        out[i][i] = ONE
    return out


def transpose(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    n, k, m = len(a), len(a[0]), len(b[0])
    out = zeros(n, m)
    for i in range(n):
        for t in range(k):
            if a[i][t].is_zero():
                continue
            for j in range(m):
                if not b[t][j].is_zero():
                    out[i][j] = out[i][j] + a[i][t] * b[t][j]
    return out


def matadd(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def matsub(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def smul(c: Q5, a: Matrix) -> Matrix:
    return [[c * x for x in row] for row in a]


def kron(a: Matrix, b: Matrix) -> Matrix:
    n, m, p, q_ = len(a), len(a[0]), len(b), len(b[0])
    out = zeros(n * p, m * q_)
    for i in range(n):
        for j in range(m):
            if a[i][j].is_zero():
                continue
            for k in range(p):
                for column in range(q_):
                    out[i * p + k][j * q_ + column] = a[i][j] * b[k][column]
    return out


def trace_matmul(a: Matrix, b: Matrix) -> Q5:
    total = ZERO
    for i in range(len(a)):
        for j in range(len(a[0])):
            total = total + a[i][j] * b[j][i]
    return total


def partial_trace_last(a: Matrix, d_last: int) -> Matrix:
    d_total = len(a)
    d_rest = d_total // d_last
    out = zeros(d_rest, d_rest)
    for i in range(d_rest):
        for j in range(d_rest):
            total = ZERO
            for k in range(d_last):
                total = total + a[i * d_last + k][j * d_last + k]
            out[i][j] = total
    return out


def partial_trace_after_first(a: Matrix, d_first: int, d_rest: int) -> Matrix:
    out = zeros(d_first, d_first)
    for i in range(d_first):
        for j in range(d_first):
            total = ZERO
            for k in range(d_rest):
                total = total + a[i * d_rest + k][j * d_rest + k]
            out[i][j] = total
    return out


def assert_matrix_equal(a: Matrix, b: Matrix, label: str) -> None:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise AssertionError(f"{label}: shape mismatch")
    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j] != b[i][j]:
                raise AssertionError(
                    f"{label}: mismatch at {(i, j)}: {a[i][j]} != {b[i][j]}"
                )


def exact_ldl_psd(a: Matrix, label: str) -> tuple[list[Q5], list[int]]:
    """Exact symmetric LDL proof with positive 1x1 pivots and zero remainder."""
    m = [row[:] for row in a]
    n = len(m)
    pivots: list[Q5] = []
    perm = list(range(n))

    def swap_sym(k: int, p: int) -> None:
        if k == p:
            return
        m[k], m[p] = m[p], m[k]
        for row in m:
            row[k], row[p] = row[p], row[k]
        perm[k], perm[p] = perm[p], perm[k]

    for k in range(n):
        candidates: list[tuple[int, float, int]] = []
        for i in range(k, n):
            sign = m[i][i].sign()
            if sign < 0:
                raise AssertionError(
                    f"{label}: negative diagonal at active row {i}: {m[i][i]}"
                )
            if sign > 0:
                candidates.append(
                    (m[i][i].complexity_bits(), -abs(m[i][i].to_float()), i)
                )

        if not candidates:
            for i in range(k, n):
                for j in range(k, n):
                    if not m[i][j].is_zero():
                        raise AssertionError(
                            f"{label}: zero diagonal remainder has nonzero {(i, j)}: {m[i][j]}"
                        )
            return pivots, perm

        _, _, p = min(candidates)
        swap_sym(k, p)
        pivot = m[k][k]
        pivots.append(pivot)

        col = [m[i][k] for i in range(k + 1, n)]
        for i in range(k + 1, n):
            multiplier = col[i - (k + 1)] / pivot
            for j in range(i, n):
                value = m[i][j] - multiplier * col[j - (k + 1)]
                m[i][j] = value
                m[j][i] = value
            m[i][k] = ZERO
            m[k][i] = ZERO

    return pivots, perm


def projector(basis: int, bit: int) -> Matrix:
    if basis == 0 and bit == 0:
        return [[ONE, ZERO], [ZERO, ZERO]]
    if basis == 0 and bit == 1:
        return [[ZERO, ZERO], [ZERO, ONE]]
    if basis == 1 and bit == 0:
        return smul(q(Fraction(1, 2)), [[ONE, ONE], [ONE, ONE]])
    if basis == 1 and bit == 1:
        return smul(q(Fraction(1, 2)), [[ONE, -ONE], [-ONE, ONE]])
    raise ValueError((basis, bit))


def build_qdb2019_tester() -> Matrix:
    tester = zeros(16, 16)
    for a in (0, 1):
        for b in (0, 1):
            j_preask = zeros(4, 4)
            for r_bit in (0, 1):
                j_preask = matadd(
                    j_preask,
                    kron(projector(a, r_bit), projector(b, r_bit)),
                )
            for c in (0, 1):
                term = kron(kron(j_preask, projector(a, c)), projector(b, c))
                tester = matadd(tester, term)
    return smul(q(Fraction(1, 8)), tester)


def build_dual_y() -> tuple[Q5, Matrix]:
    mu = (q(5) + SQRT5) / q(8)
    high = q(Fraction(3, 10)) * mu
    low = q(Fraction(1, 5)) * mu
    off = q(Fraction(1, 20)) * mu
    y = zeros(8, 8)
    for idx in (0, 1, 6, 7):
        y[idx][idx] = high
    for idx in (2, 3, 4, 5):
        y[idx][idx] = low
    signed_pairs = {
        (0, 2): 1,
        (0, 4): 1,
        (0, 5): 1,
        (0, 6): 1,
        (1, 3): 1,
        (1, 4): -1,
        (1, 5): 1,
        (1, 7): 1,
        (2, 4): 1,
        (2, 6): -1,
        (2, 7): 1,
        (3, 5): 1,
        (3, 6): -1,
        (3, 7): -1,
        (4, 6): -1,
        (5, 7): -1,
    }
    for (i, j), sign in signed_pairs.items():
        y[i][j] = y[j][i] = q(sign) * off
    return mu, y


def build_kernel_basis() -> Matrix:
    """Columns span ker(Y otimes I - Wbar), in [Q,R,C,O] lexicographic order."""
    k = zeros(16, 4)
    entries = [
        [
            (0, q(Fraction(3, 2), Fraction(-1, 2))),
            (1, q(Fraction(-3, 2), Fraction(1, 2))),
            (2, q(Fraction(-3, 2), Fraction(1, 2))),
            (3, q(Fraction(-1, 2), Fraction(1, 2))),
            (4, q(Fraction(-1, 2), Fraction(1, 2))),
            (5, q(Fraction(-1, 2), Fraction(1, 2))),
            (6, q(Fraction(3, 2), Fraction(-1, 2))),
            (7, q(Fraction(1, 2), Fraction(-1, 2))),
            (8, q(-1)),
            (9, q(1, -1)),
            (10, q(-1)),
            (11, q(1)),
            (12, q(1)),
        ],
        [
            (0, q(Fraction(-3, 2), Fraction(-3, 2))),
            (1, q(Fraction(-3, 2), Fraction(1, 2))),
            (2, q(Fraction(-1, 2), Fraction(1, 2))),
            (3, q(Fraction(-5, 2), Fraction(-1, 2))),
            (4, q(Fraction(-1, 2), Fraction(1, 2))),
            (5, q(Fraction(-3, 2), Fraction(-1, 2))),
            (6, q(Fraction(1, 2), Fraction(-1, 2))),
            (7, q(Fraction(5, 2), Fraction(1, 2))),
            (8, q(1, 1)),
            (9, q(1)),
            (10, q(1)),
            (11, q(1)),
            (13, q(1)),
        ],
        [
            (0, q(Fraction(5, 2), Fraction(1, 2))),
            (1, q(Fraction(-1, 2), Fraction(1, 2))),
            (2, q(Fraction(-3, 2), Fraction(1, 2))),
            (3, q(Fraction(3, 2), Fraction(3, 2))),
            (4, q(Fraction(-5, 2), Fraction(-1, 2))),
            (5, q(Fraction(1, 2), Fraction(-1, 2))),
            (6, q(Fraction(-3, 2), Fraction(-1, 2))),
            (7, q(Fraction(1, 2), Fraction(-1, 2))),
            (8, q(-1)),
            (9, q(1)),
            (10, q(1)),
            (11, q(-1, -1)),
            (14, q(1)),
        ],
        [
            (0, q(Fraction(-1, 2), Fraction(1, 2))),
            (1, q(Fraction(3, 2), Fraction(-1, 2))),
            (2, q(Fraction(3, 2), Fraction(-1, 2))),
            (3, q(Fraction(3, 2), Fraction(-1, 2))),
            (4, q(Fraction(1, 2), Fraction(-1, 2))),
            (5, q(Fraction(-3, 2), Fraction(1, 2))),
            (6, q(Fraction(1, 2), Fraction(-1, 2))),
            (7, q(Fraction(-1, 2), Fraction(1, 2))),
            (8, q(1)),
            (9, q(1)),
            (10, q(-1, 1)),
            (11, q(-1)),
            (15, q(1)),
        ],
    ]
    for col, col_entries in enumerate(entries):
        for row, value in col_entries:
            k[row][col] = value
    return k


def build_primal_m() -> Matrix:
    return [
        [
            q(Fraction(5, 16), Fraction(1, 16)),
            q(Fraction(-1, 16), Fraction(1, 16)),
            q(Fraction(-1, 16), Fraction(3, 80)),
            q(Fraction(1, 16), Fraction(7, 80)),
        ],
        [
            q(Fraction(-1, 16), Fraction(1, 16)),
            q(Fraction(3, 16), Fraction(-1, 16)),
            q(Fraction(1, 16), Fraction(-1, 80)),
            q(Fraction(1, 16), Fraction(-3, 80)),
        ],
        [
            q(Fraction(-1, 16), Fraction(3, 80)),
            q(Fraction(1, 16), Fraction(-1, 80)),
            q(Fraction(3, 16), Fraction(-1, 16)),
            q(Fraction(1, 16), Fraction(-1, 16)),
        ],
        [
            q(Fraction(1, 16), Fraction(7, 80)),
            q(Fraction(1, 16), Fraction(-3, 80)),
            q(Fraction(1, 16), Fraction(-1, 16)),
            q(Fraction(5, 16), Fraction(1, 16)),
        ],
    ]


def main() -> None:
    tester = build_qdb2019_tester()
    mu, y = build_dual_y()
    eta = smul(q(Fraction(1, 2)), eye(2))

    dual_slack_strategy = matsub(kron(y, eye(2)), tester)
    dual_slack_normalization = matsub(
        smul(mu, eye(2)),
        partial_trace_after_first(y, d_first=2, d_rest=4),
    )
    assert_matrix_equal(
        dual_slack_normalization, zeros(2, 2), "dual normalization slack"
    )
    dual_pivots, dual_perm = exact_ldl_psd(dual_slack_strategy, "dual strategy slack")

    k = build_kernel_basis()
    assert_matrix_equal(
        matmul(dual_slack_strategy, k), zeros(16, 4), "dual slack kernel"
    )

    m = build_primal_m()
    primal_m_pivots, primal_m_perm = exact_ldl_psd(m, "primal kernel coefficient M")
    s = matmul(matmul(k, m), transpose(k))
    assert_matrix_equal(
        partial_trace_last(s, d_last=2), kron(eta, eye(4)), "comb normalization"
    )

    primal_value = trace_matmul(tester, s)
    if primal_value != mu:
        raise AssertionError(f"objective mismatch: {primal_value} != {mu}")

    print("QDB 2019 exact one-round MF witness")
    print(f"mu = {mu} ~= {mu.to_float():.15f}")
    print()
    print("Dual verification")
    print("  normalization slack: exactly zero")
    print(f"  strategy slack: exact LDL with {len(dual_pivots)} positive pivots")
    print(f"  strategy slack permutation: {dual_perm}")
    print("  strategy slack pivots:")
    for i, pivot in enumerate(dual_pivots):
        print(f"    d[{i}] = {pivot} ~= {pivot.to_float():.12f}")
    print()
    print("Primal verification")
    print("  eta = I/2")
    print("  S = K M K^T with A K = 0 exactly")
    print(f"  M: exact LDL with {len(primal_m_pivots)} positive pivots")
    print(f"  M permutation: {primal_m_perm}")
    print("  M pivots:")
    for i, pivot in enumerate(primal_m_pivots):
        print(f"    d[{i}] = {pivot} ~= {pivot.to_float():.12f}")
    print("  Tr_O(S) = eta tensor I_RC exactly")
    print(f"  Tr(Wbar S) = {primal_value} ~= {primal_value.to_float():.15f}")
    print()
    print("Conclusion: primal value = dual upper bound = (5 + sqrt(5)) / 8.")


if __name__ == "__main__":
    main()
