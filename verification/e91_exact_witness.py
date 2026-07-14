#!/usr/bin/env python3
"""Exact E91 QDB one-round MF witness in a degree-four number field.

This script verifies the closed-form primal/dual certificate

    p_MF^*(E91 retained) = 1/2 + sqrt(24 + 17 sqrt(2)) / 16

for the retained value-check one-round lifted-comb SDP used by the benchmark.
All matrix identities use exact arithmetic in

    Q(sqrt(2), sqrt(24 + 17 sqrt(2))).

Positivity is checked with exact LDL decompositions. Signs of field elements
are decided using successively refined rational enclosures of both positive
square roots, not floating-point approximations. No numerical SDP solver or
stored numerical certificate is used.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction

sys.set_int_max_str_digits(0)


Q2Pair = tuple[Fraction, Fraction]


def _q2_add(left: Q2Pair, right: Q2Pair) -> Q2Pair:
    return left[0] + right[0], left[1] + right[1]


def _q2_neg(value: Q2Pair) -> Q2Pair:
    return -value[0], -value[1]


def _q2_mul(left: Q2Pair, right: Q2Pair) -> Q2Pair:
    return (
        left[0] * right[0] + 2 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _q2_inv(value: Q2Pair) -> Q2Pair:
    denominator = value[0] * value[0] - 2 * value[1] * value[1]
    if denominator == 0:
        raise ZeroDivisionError("zero in Q(sqrt(2))")
    return value[0] / denominator, -value[1] / denominator


@dataclass(frozen=True)
class Q4:
    """Element a + b*r + c*s + d*r*s.

    Here r = sqrt(2) and s = sqrt(24 + 17*r), both under their positive
    real embeddings.
    """

    a: Fraction = Fraction(0)
    b: Fraction = Fraction(0)
    c: Fraction = Fraction(0)
    d: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", Fraction(self.a))
        object.__setattr__(self, "b", Fraction(self.b))
        object.__setattr__(self, "c", Fraction(self.c))
        object.__setattr__(self, "d", Fraction(self.d))

    def __add__(self, other: object) -> Q4:
        other = q4(other)
        return Q4(
            self.a + other.a,
            self.b + other.b,
            self.c + other.c,
            self.d + other.d,
        )

    __radd__ = __add__

    def __neg__(self) -> Q4:
        return Q4(-self.a, -self.b, -self.c, -self.d)

    def __sub__(self, other: object) -> Q4:
        return self + (-q4(other))

    def __rsub__(self, other: object) -> Q4:
        return q4(other) - self

    def __mul__(self, other: object) -> Q4:
        other = q4(other)
        left_a = (self.a, self.b)
        left_b = (self.c, self.d)
        right_a = (other.a, other.b)
        right_b = (other.c, other.d)
        radical_square = (Fraction(24), Fraction(17))

        constant = _q2_add(
            _q2_mul(left_a, right_a),
            _q2_mul(_q2_mul(left_b, right_b), radical_square),
        )
        radical = _q2_add(
            _q2_mul(left_a, right_b),
            _q2_mul(left_b, right_a),
        )
        return Q4(constant[0], constant[1], radical[0], radical[1])

    __rmul__ = __mul__

    def inv(self) -> Q4:
        constant = (self.a, self.b)
        radical = (self.c, self.d)
        radical_square = (Fraction(24), Fraction(17))
        denominator = _q2_add(
            _q2_mul(constant, constant),
            _q2_neg(_q2_mul(_q2_mul(radical, radical), radical_square)),
        )
        denominator_inv = _q2_inv(denominator)
        numerator_constant = _q2_mul(denominator_inv, constant)
        numerator_radical = _q2_neg(_q2_mul(denominator_inv, radical))
        return Q4(
            numerator_constant[0],
            numerator_constant[1],
            numerator_radical[0],
            numerator_radical[1],
        )

    def __truediv__(self, other: object) -> Q4:
        return self * q4(other).inv()

    def __rtruediv__(self, other: object) -> Q4:
        return q4(other) * self.inv()

    def is_zero(self) -> bool:
        return self.a == self.b == self.c == self.d == 0

    def sign(self) -> int:
        """Return the exact sign under the positive real embedding."""
        if self.is_zero():
            return 0

        for bits in (16, 32, 64, 128, 256, 512, 1024, 2048):
            lower, upper = self._rational_interval(1 << bits)
            if lower > 0:
                return 1
            if upper < 0:
                return -1
        raise ArithmeticError(f"failed to separate algebraic number from zero: {self}")

    def _rational_interval(self, scale: int) -> tuple[Fraction, Fraction]:
        root2_lower, root2_upper = _sqrt_fraction_interval(Fraction(2), scale)
        alpha_square_lower = 24 + 17 * root2_lower
        alpha_square_upper = 24 + 17 * root2_upper
        alpha_lower, _ = _sqrt_fraction_interval(alpha_square_lower, scale)
        _, alpha_upper = _sqrt_fraction_interval(alpha_square_upper, scale)
        root_product_lower = root2_lower * alpha_lower
        root_product_upper = root2_upper * alpha_upper

        lower = upper = self.a
        for coefficient, interval in (
            (self.b, (root2_lower, root2_upper)),
            (self.c, (alpha_lower, alpha_upper)),
            (self.d, (root_product_lower, root_product_upper)),
        ):
            if coefficient >= 0:
                lower += coefficient * interval[0]
                upper += coefficient * interval[1]
            else:
                lower += coefficient * interval[1]
                upper += coefficient * interval[0]
        return lower, upper

    def to_float(self) -> float:
        root2 = math.sqrt(2.0)
        alpha = math.sqrt(24.0 + 17.0 * root2)
        return (
            float(self.a)
            + float(self.b) * root2
            + float(self.c) * alpha
            + float(self.d) * root2 * alpha
        )

    def __str__(self) -> str:
        terms: list[str] = []
        for coefficient, suffix in (
            (self.a, ""),
            (self.b, "*sqrt2"),
            (self.c, "*alpha"),
            (self.d, "*sqrt2*alpha"),
        ):
            if coefficient:
                terms.append(f"{coefficient}{suffix}")
        return " + ".join(terms).replace("+ -", "- ") if terms else "0"


def _sqrt_fraction_interval(
    value: Fraction,
    scale: int,
) -> tuple[Fraction, Fraction]:
    if value <= 0:
        raise ValueError("square-root enclosure requires a positive value")
    scaled_floor = (value.numerator * scale * scale) // value.denominator
    lower_numerator = math.isqrt(scaled_floor)
    lower = Fraction(lower_numerator, scale)
    upper = Fraction(lower_numerator + 1, scale)
    if not lower * lower <= value < upper * upper:
        raise AssertionError("invalid rational square-root enclosure")
    return lower, upper


def q4(value: object) -> Q4:
    if isinstance(value, Q4):
        return value
    return Q4(Fraction(value))


def q(
    a: int | Fraction = 0,
    b: int | Fraction = 0,
    c: int | Fraction = 0,
    d: int | Fraction = 0,
) -> Q4:
    return Q4(Fraction(a), Fraction(b), Fraction(c), Fraction(d))


ZERO = Q4()
ONE = Q4(1)
ROOT2 = Q4(0, 1)
ALPHA = Q4(0, 0, 1)
ROOT2_ALPHA = Q4(0, 0, 0, 1)

Matrix = list[list[Q4]]


def zeros(rows: int, columns: int) -> Matrix:
    return [[ZERO for _ in range(columns)] for _ in range(rows)]


def eye(size: int) -> Matrix:
    out = zeros(size, size)
    for index in range(size):
        out[index][index] = ONE
    return out


def transpose(matrix: Matrix) -> Matrix:
    return [list(row) for row in zip(*matrix)]


def matadd(left: Matrix, right: Matrix) -> Matrix:
    return [
        [left[i][j] + right[i][j] for j in range(len(left[0]))]
        for i in range(len(left))
    ]


def matsub(left: Matrix, right: Matrix) -> Matrix:
    return [
        [left[i][j] - right[i][j] for j in range(len(left[0]))]
        for i in range(len(left))
    ]


def smul(coefficient: Q4, matrix: Matrix) -> Matrix:
    return [[coefficient * value for value in row] for row in matrix]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    out = zeros(len(left), len(right[0]))
    for i in range(len(left)):
        for k in range(len(right)):
            if left[i][k].is_zero():
                continue
            for j in range(len(right[0])):
                if not right[k][j].is_zero():
                    out[i][j] = out[i][j] + left[i][k] * right[k][j]
    return out


def kron(left: Matrix, right: Matrix) -> Matrix:
    out = zeros(
        len(left) * len(right),
        len(left[0]) * len(right[0]),
    )
    for i in range(len(left)):
        for j in range(len(left[0])):
            if left[i][j].is_zero():
                continue
            for k in range(len(right)):
                for ell in range(len(right[0])):
                    out[i * len(right) + k][j * len(right[0]) + ell] = (
                        left[i][j] * right[k][ell]
                    )
    return out


def trace_matmul(left: Matrix, right: Matrix) -> Q4:
    total = ZERO
    for i in range(len(left)):
        for j in range(len(left[0])):
            total = total + left[i][j] * right[j][i]
    return total


def partial_trace_last(matrix: Matrix, last_dimension: int) -> Matrix:
    rest_dimension = len(matrix) // last_dimension
    out = zeros(rest_dimension, rest_dimension)
    for i in range(rest_dimension):
        for j in range(rest_dimension):
            out[i][j] = sum(
                (
                    matrix[i * last_dimension + k][j * last_dimension + k]
                    for k in range(last_dimension)
                ),
                ZERO,
            )
    return out


def partial_trace_after_first(
    matrix: Matrix,
    first_dimension: int,
    rest_dimension: int,
) -> Matrix:
    out = zeros(first_dimension, first_dimension)
    for i in range(first_dimension):
        for j in range(first_dimension):
            out[i][j] = sum(
                (
                    matrix[i * rest_dimension + k][j * rest_dimension + k]
                    for k in range(rest_dimension)
                ),
                ZERO,
            )
    return out


def extract(matrix: Matrix, indices: list[int]) -> Matrix:
    return [[matrix[i][j] for j in indices] for i in indices]


def assert_matrix_equal(left: Matrix, right: Matrix, label: str) -> None:
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise AssertionError(f"{label}: shape mismatch")
    for i in range(len(left)):
        for j in range(len(left[0])):
            if left[i][j] != right[i][j]:
                raise AssertionError(
                    f"{label}: mismatch at {(i, j)}: {left[i][j]} != {right[i][j]}"
                )


def exact_ldl_psd(matrix: Matrix, label: str) -> int:
    """Prove a real symmetric matrix PSD by exact pivoted LDL."""
    if len(matrix) != len(matrix[0]):
        raise AssertionError(f"{label}: matrix is not square")
    assert_matrix_equal(matrix, transpose(matrix), f"{label} symmetry")

    active = [row[:] for row in matrix]
    size = len(active)
    pivot_count = 0

    def swap_symmetric(left_index: int, right_index: int) -> None:
        if left_index == right_index:
            return
        active[left_index], active[right_index] = (
            active[right_index],
            active[left_index],
        )
        for row in active:
            row[left_index], row[right_index] = (
                row[right_index],
                row[left_index],
            )

    for k in range(size):
        positive_index: int | None = None
        for i in range(k, size):
            diagonal_sign = active[i][i].sign()
            if diagonal_sign < 0:
                raise AssertionError(
                    f"{label}: negative active diagonal {i}: {active[i][i]}"
                )
            if diagonal_sign > 0 and positive_index is None:
                positive_index = i

        if positive_index is None:
            for i in range(k, size):
                for j in range(k, size):
                    if not active[i][j].is_zero():
                        raise AssertionError(
                            f"{label}: zero-diagonal remainder has nonzero {(i, j)}"
                        )
            return pivot_count

        swap_symmetric(k, positive_index)
        pivot = active[k][k]
        column = [active[i][k] for i in range(k + 1, size)]
        for i in range(k + 1, size):
            multiplier = column[i - k - 1] / pivot
            for j in range(i, size):
                value = active[i][j] - multiplier * column[j - k - 1]
                active[i][j] = value
                active[j][i] = value
            active[i][k] = ZERO
            active[k][i] = ZERO
        pivot_count += 1

    return pivot_count


def nullspace(matrix: Matrix) -> Matrix:
    """Return an exact column basis for the nullspace."""
    reduced = [row[:] for row in matrix]
    row_count = len(reduced)
    column_count = len(reduced[0])
    pivot_columns: list[int] = []
    pivot_row = 0

    for column in range(column_count):
        selected = next(
            (
                row
                for row in range(pivot_row, row_count)
                if not reduced[row][column].is_zero()
            ),
            None,
        )
        if selected is None:
            continue
        reduced[pivot_row], reduced[selected] = reduced[selected], reduced[pivot_row]
        pivot = reduced[pivot_row][column]
        reduced[pivot_row] = [entry / pivot for entry in reduced[pivot_row]]
        for row in range(row_count):
            if row == pivot_row or reduced[row][column].is_zero():
                continue
            multiplier = reduced[row][column]
            reduced[row] = [
                reduced[row][j] - multiplier * reduced[pivot_row][j]
                for j in range(column_count)
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break

    free_columns = [
        column for column in range(column_count) if column not in pivot_columns
    ]
    basis = zeros(column_count, len(free_columns))
    for basis_column, free_column in enumerate(free_columns):
        basis[free_column][basis_column] = ONE
        for row, pivot_column in enumerate(pivot_columns):
            basis[pivot_column][basis_column] = -reduced[row][free_column]
    assert_matrix_equal(
        matmul(matrix, basis),
        zeros(row_count, len(free_columns)),
        "nullspace",
    )
    return basis


def projector(basis: int, bit: int) -> Matrix:
    """Exact Z, X, and W rank-one projectors."""
    if basis == 0:
        if bit == 0:
            return [[ONE, ZERO], [ZERO, ZERO]]
        return [[ZERO, ZERO], [ZERO, ONE]]
    if basis == 1:
        sign = ONE if bit == 0 else -ONE
        return smul(q(Fraction(1, 2)), [[ONE, sign], [sign, ONE]])
    if basis == 2:
        root_term = ROOT2 / q(4)
        if bit == 0:
            return [
                [q(Fraction(1, 2)) + root_term, root_term],
                [root_term, q(Fraction(1, 2)) - root_term],
            ]
        return [
            [q(Fraction(1, 2)) - root_term, -root_term],
            [-root_term, q(Fraction(1, 2)) + root_term],
        ]
    raise ValueError(f"unsupported E91 basis: {basis}")


def build_e91_tester() -> Matrix:
    tester = zeros(16, 16)
    branch_count = 0
    # Retained pairs: (a_v,a_p)=(1,0) is W/W; (2,1) is Z/Z.
    for verifier_basis, prover_basis in ((2, 2), (0, 0)):
        for response_basis in (0, 1):
            preask = zeros(4, 4)
            for outcome in (0, 1):
                preask = matadd(
                    preask,
                    kron(
                        projector(prover_basis, outcome),
                        projector(response_basis, outcome),
                    ),
                )
            for verifier_outcome in (0, 1):
                target_bit = 1 - verifier_outcome
                term = kron(
                    kron(preask, projector(verifier_basis, target_bit)),
                    projector(response_basis, target_bit),
                )
                tester = matadd(tester, term)
                branch_count += 1
    if branch_count != 8:
        raise AssertionError(f"unexpected E91 branch count: {branch_count}")
    return smul(q(Fraction(1, branch_count)), tester)


def signed_pattern(
    entries: list[tuple[int, int, int]],
) -> Matrix:
    pattern = zeros(8, 8)
    for row, column, sign in entries:
        pattern[row][column] = q(sign)
    return pattern


def build_dual_y() -> tuple[Q4, Matrix]:
    patterns = [
        signed_pattern([(0, 0, 1), (1, 1, 1), (6, 6, 1), (7, 7, 1)]),
        signed_pattern([(2, 2, 1), (3, 3, 1), (4, 4, 1), (5, 5, 1)]),
        signed_pattern(
            [
                (0, 2, 1),
                (1, 3, 1),
                (2, 0, 1),
                (3, 1, 1),
                (4, 6, -1),
                (5, 7, -1),
                (6, 4, -1),
                (7, 5, -1),
            ]
        ),
        signed_pattern(
            [
                (0, 4, 1),
                (0, 6, 1),
                (1, 5, 1),
                (1, 7, 1),
                (2, 4, 1),
                (2, 6, -1),
                (3, 5, 1),
                (3, 7, -1),
                (4, 0, 1),
                (4, 2, 1),
                (5, 1, 1),
                (5, 3, 1),
                (6, 0, 1),
                (6, 2, -1),
                (7, 1, 1),
                (7, 3, -1),
            ]
        ),
        signed_pattern(
            [
                (0, 5, 1),
                (1, 4, -1),
                (2, 7, 1),
                (3, 6, -1),
                (4, 1, -1),
                (5, 0, 1),
                (6, 3, -1),
                (7, 2, 1),
            ]
        ),
    ]
    coefficients = [
        (q(10) + ROOT2 + ROOT2_ALPHA) / q(64),
        (q(6) - ROOT2 + 2 * ALPHA - ROOT2_ALPHA) / q(64),
        (q(2) + ROOT2 - ALPHA + ROOT2_ALPHA) / q(64),
        (ROOT2 + 3 * ALPHA - 2 * ROOT2_ALPHA) / q(64),
        (3 * ALPHA - 2 * ROOT2_ALPHA) / q(64),
    ]
    dual = zeros(8, 8)
    for coefficient, pattern in zip(coefficients, patterns):
        dual = matadd(dual, smul(coefficient, pattern))
    value = q(Fraction(1, 2)) + ALPHA / q(16)
    return value, dual


def bell_transform() -> tuple[Matrix, dict[tuple[int, int], list[int]]]:
    unitary = smul(
        ROOT2 / q(2),
        [
            [ONE, ONE, ZERO, ZERO],
            [ZERO, ZERO, ONE, ONE],
            [ZERO, ZERO, ONE, -ONE],
            [ONE, -ONE, ZERO, ZERO],
        ],
    )
    transform = kron(unitary, unitary)
    assert_matrix_equal(
        matmul(transpose(transform), transform),
        eye(16),
        "Bell transform orthogonality",
    )

    labels_one = [-1, 1, 1, -1]
    labels = [(left, right) for left in labels_one for right in labels_one]
    groups = {
        key: [index for index, label in enumerate(labels) if label == key]
        for key in sorted(set(labels))
    }
    return transform, groups


def primal_kernel_coefficients() -> list[Matrix]:
    return [
        [
            [
                q(
                    Fraction(507144, 2627520),
                    Fraction(128010, 2627520),
                    Fraction(-971652, 2627520),
                    Fraction(670021, 2627520),
                ),
                q(
                    Fraction(45900, 2627520),
                    Fraction(-64702, 2627520),
                    Fraction(-1852320, 2627520),
                    Fraction(1331151, 2627520),
                ),
            ],
            [
                q(
                    Fraction(45900, 2627520),
                    Fraction(-64702, 2627520),
                    Fraction(-1852320, 2627520),
                    Fraction(1331151, 2627520),
                ),
                q(
                    Fraction(331160, 2627520),
                    Fraction(86258, 2627520),
                    Fraction(499324, 2627520),
                    Fraction(-314211, 2627520),
                ),
            ],
        ],
        [
            [
                q(
                    Fraction(19856, 57120),
                    Fraction(-4658, 57120),
                    Fraction(10268, 57120),
                    Fraction(-7735, 57120),
                ),
                q(Fraction(-1, 120)),
            ],
            [q(Fraction(-1, 120)), q(Fraction(1, 3))],
        ],
        [
            [
                q(
                    Fraction(25058, 129472),
                    Fraction(14773, 129472),
                    Fraction(-15181, 129472),
                    Fraction(8789, 129472),
                ),
                q(
                    Fraction(-24786, 129472),
                    Fraction(-16847, 129472),
                    Fraction(49657, 129472),
                    Fraction(-27897, 129472),
                ),
            ],
            [
                q(
                    Fraction(-24786, 129472),
                    Fraction(-16847, 129472),
                    Fraction(49657, 129472),
                    Fraction(-27897, 129472),
                ),
                q(
                    Fraction(24650, 129472),
                    Fraction(7769, 129472),
                    Fraction(9979, 129472),
                    Fraction(-7055, 129472),
                ),
            ],
        ],
        [
            [
                q(
                    Fraction(680, 3808),
                    Fraction(-663, 3808),
                    Fraction(153, 3808),
                    Fraction(-51, 3808),
                ),
                q(Fraction(1, 8)),
            ],
            [q(Fraction(1, 8)), q(Fraction(7, 16))],
        ],
    ]


def main() -> None:
    tester = build_e91_tester()
    value, dual_y = build_dual_y()
    transform, groups = bell_transform()

    dual_normalization = matsub(
        smul(value, eye(2)),
        partial_trace_after_first(dual_y, first_dimension=2, rest_dimension=4),
    )
    assert_matrix_equal(
        dual_normalization,
        zeros(2, 2),
        "dual normalization slack",
    )

    dual_strategy = matsub(kron(dual_y, eye(2)), tester)
    transformed_dual = matmul(
        matmul(transpose(transform), dual_strategy),
        transform,
    )
    index_to_group = {
        index: key for key, indices in groups.items() for index in indices
    }
    for i in range(16):
        for j in range(16):
            if (
                index_to_group[i] != index_to_group[j]
                and not transformed_dual[i][j].is_zero()
            ):
                raise AssertionError(
                    f"transformed dual is not block diagonal at {(i, j)}"
                )

    dual_ranks: list[int] = []
    kernel_bases: list[Matrix] = []
    for key, indices in groups.items():
        block = extract(transformed_dual, indices)
        dual_ranks.append(exact_ldl_psd(block, f"dual block {key}"))
        kernel = nullspace(block)
        if len(kernel[0]) != 2:
            raise AssertionError(f"dual block {key}: expected nullity two")
        kernel_bases.append(kernel)

    coefficient_matrices = primal_kernel_coefficients()
    transformed_primal = zeros(16, 16)
    primal_coefficient_ranks: list[int] = []
    for (key, indices), kernel, coefficient in zip(
        groups.items(),
        kernel_bases,
        coefficient_matrices,
    ):
        rank = exact_ldl_psd(coefficient, f"primal coefficient {key}")
        if rank != 2:
            raise AssertionError(
                f"primal coefficient {key}: expected positive definite"
            )
        primal_coefficient_ranks.append(rank)
        block = matmul(matmul(kernel, coefficient), transpose(kernel))
        for local_i, global_i in enumerate(indices):
            for local_j, global_j in enumerate(indices):
                transformed_primal[global_i][global_j] = block[local_i][local_j]

    primal = matmul(
        matmul(transform, transformed_primal),
        transpose(transform),
    )
    eta = smul(q(Fraction(1, 2)), eye(2))
    assert_matrix_equal(
        partial_trace_last(primal, last_dimension=2),
        kron(eta, eye(4)),
        "primal comb normalization",
    )
    if eta[0][0] + eta[1][1] != ONE:
        raise AssertionError("trace(eta) is not one")

    primal_value = trace_matmul(tester, primal)
    if primal_value != value:
        raise AssertionError(f"objective mismatch: {primal_value} != {value}")
    complementary_product = trace_matmul(dual_strategy, primal)
    if not complementary_product.is_zero():
        raise AssertionError(f"complementary slackness failed: {complementary_product}")

    print("E91 QDB retained value-check exact one-round MF witness")
    print("register order: Q,R,C,O")
    print("branch distribution: uniform over 8 retained (a_v,a_p,b,m) branches")
    print("number field: Q(sqrt(2), alpha), alpha=sqrt(24+17*sqrt(2))")
    print(f"value: 1/2 + alpha/16 ~= {value.to_float():.15f}")
    print()
    print("Dual verification")
    print("  Tr_RC(Y) = value * I_Q exactly")
    print("  Y tensor I_O - Wbar splits into four exact parity blocks")
    print(f"  exact LDL positive-pivot counts: {dual_ranks}")
    print()
    print("Primal verification")
    print("  eta = I/2")
    print("  each parity block is K A K^T on the dual kernel")
    print(f"  exact LDL ranks of A: {primal_coefficient_ranks}")
    print("  Tr_O(S) = eta tensor I_RC exactly")
    print(f"  Tr(Wbar S) = value ~= {primal_value.to_float():.15f}")
    print("  complementary slackness holds exactly")
    print()
    print(
        "Conclusion: primal value = dual upper bound = "
        "(8 + sqrt(24 + 17 sqrt(2))) / 16."
    )


if __name__ == "__main__":
    main()
