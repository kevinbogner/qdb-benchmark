#!/usr/bin/env python3
"""Exact rational MF witnesses for QDB 2017 and Mutual QDB.

The two reduced one-round lifted-comb SDPs have rational testers and rational
matching primal/dual certificates:

    p_MF^*(QDB 2017 rapid phase) = 1
    p_MF^*(Mutual QDB first timed return leg) = 3/4.

All checks below use fractions only. No numerical eigensolver, SDP solver, or
stored numerical certificate is used.
"""

from __future__ import annotations

import argparse
from fractions import Fraction

Scalar = Fraction
Matrix = list[list[Scalar]]

ZERO = Fraction(0)
ONE = Fraction(1)


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


def smul(coefficient: Scalar, matrix: Matrix) -> Matrix:
    return [[coefficient * value for value in row] for row in matrix]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    out = zeros(len(left), len(right[0]))
    for i in range(len(left)):
        for k in range(len(right)):
            if left[i][k] == 0:
                continue
            for j in range(len(right[0])):
                if right[k][j] != 0:
                    out[i][j] += left[i][k] * right[k][j]
    return out


def kron(left: Matrix, right: Matrix) -> Matrix:
    out = zeros(
        len(left) * len(right),
        len(left[0]) * len(right[0]),
    )
    for i in range(len(left)):
        for j in range(len(left[0])):
            if left[i][j] == 0:
                continue
            for k in range(len(right)):
                for ell in range(len(right[0])):
                    out[i * len(right) + k][j * len(right[0]) + ell] = (
                        left[i][j] * right[k][ell]
                    )
    return out


def trace_matmul(left: Matrix, right: Matrix) -> Scalar:
    return sum(
        (
            left[i][j] * right[j][i]
            for i in range(len(left))
            for j in range(len(left[0]))
        ),
        ZERO,
    )


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


def assert_matrix_equal(left: Matrix, right: Matrix, label: str) -> None:
    if left != right:
        for i in range(len(left)):
            for j in range(len(left[0])):
                if left[i][j] != right[i][j]:
                    raise AssertionError(
                        f"{label}: mismatch at {(i, j)}: {left[i][j]} != {right[i][j]}"
                    )
        raise AssertionError(f"{label}: shape mismatch")


def exact_ldl_psd(matrix: Matrix, label: str) -> int:
    """Prove a rational symmetric matrix PSD by exact pivoted LDL."""
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
            if active[i][i] < 0:
                raise AssertionError(
                    f"{label}: negative active diagonal {i}: {active[i][i]}"
                )
            if active[i][i] > 0 and positive_index is None:
                positive_index = i

        if positive_index is None:
            for i in range(k, size):
                for j in range(k, size):
                    if active[i][j] != 0:
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


def projector(basis: int, bit: int) -> Matrix:
    if basis == 0:
        if bit == 0:
            return [[ONE, ZERO], [ZERO, ZERO]]
        return [[ZERO, ZERO], [ZERO, ONE]]
    if basis == 1:
        sign = ONE if bit == 0 else -ONE
        return smul(Fraction(1, 2), [[ONE, sign], [sign, ONE]])
    raise ValueError(f"unsupported BB84 basis: {basis}")


def build_qdb2017_tester() -> Matrix:
    tester = zeros(16, 16)
    branch_count = 0
    for basis in (0, 1):
        preask = zeros(4, 4)
        for outcome in (0, 1):
            preask = matadd(
                preask,
                kron(projector(basis, outcome), projector(basis, outcome)),
            )
        for challenge in (0, 1):
            term = kron(
                kron(preask, projector(basis, challenge)),
                projector(basis, challenge),
            )
            tester = matadd(tester, term)
            branch_count += 1
    return smul(Fraction(1, branch_count), tester)


def build_mutual_tester() -> Matrix:
    tester = zeros(16, 16)
    branch_count = 0
    for challenge_basis in (0, 1):
        for response_basis in (0, 1):
            for mask in (0, 1):
                preask = zeros(4, 4)
                for outcome in (0, 1):
                    preask = matadd(
                        preask,
                        kron(
                            projector(challenge_basis, outcome),
                            projector(response_basis, outcome ^ mask),
                        ),
                    )
                for verifier_outcome in (0, 1):
                    term = kron(
                        kron(
                            preask,
                            projector(challenge_basis, verifier_outcome),
                        ),
                        projector(
                            response_basis,
                            verifier_outcome ^ mask,
                        ),
                    )
                    tester = matadd(tester, term)
                    branch_count += 1
    if branch_count != 16:
        raise AssertionError(f"unexpected Mutual QDB branch count: {branch_count}")
    return smul(Fraction(1, branch_count), tester)


def verify_qdb2017() -> None:
    tester = build_qdb2017_tester()
    value = ONE
    dual_y = [
        [Fraction(3, 8), 0, 0, 0, 0, 0, Fraction(1, 8), 0],
        [0, Fraction(3, 8), 0, 0, 0, 0, 0, Fraction(1, 8)],
        [0, 0, Fraction(1, 8), 0, Fraction(1, 8), 0, 0, 0],
        [0, 0, 0, Fraction(1, 8), 0, Fraction(1, 8), 0, 0],
        [0, 0, Fraction(1, 8), 0, Fraction(1, 8), 0, 0, 0],
        [0, 0, 0, Fraction(1, 8), 0, Fraction(1, 8), 0, 0],
        [Fraction(1, 8), 0, 0, 0, 0, 0, Fraction(3, 8), 0],
        [0, Fraction(1, 8), 0, 0, 0, 0, 0, Fraction(3, 8)],
    ]
    dual_normalization = matsub(
        smul(value, eye(2)),
        partial_trace_after_first(dual_y, first_dimension=2, rest_dimension=4),
    )
    assert_matrix_equal(
        dual_normalization,
        zeros(2, 2),
        "QDB 2017 dual normalization",
    )
    dual_strategy = matsub(kron(dual_y, eye(2)), tester)
    dual_rank = exact_ldl_psd(dual_strategy, "QDB 2017 dual strategy slack")

    eta = smul(Fraction(1, 2), eye(2))
    identity_choi = [
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
    ]
    primal = kron(kron(eta, eye(2)), identity_choi)
    primal_rank = exact_ldl_psd(primal, "QDB 2017 primal comb")
    assert_matrix_equal(
        partial_trace_last(primal, last_dimension=2),
        kron(eta, eye(4)),
        "QDB 2017 comb normalization",
    )
    primal_value = trace_matmul(tester, primal)
    if primal_value != value:
        raise AssertionError(f"QDB 2017 objective mismatch: {primal_value}")

    print("QDB 2017 rapid-phase exact one-round MF witness")
    print(f"  primal value = dual upper bound = {value}")
    print(f"  dual slack exact LDL rank: {dual_rank}")
    print(f"  primal comb exact LDL rank: {primal_rank}")
    print("  Tr_O(S) = eta tensor I_RC exactly, eta=I/2")


def mutual_dual_y() -> Matrix:
    dual = smul(Fraction(3, 16), eye(8))
    for row, column, sign in (
        (0, 5, 1),
        (1, 4, -1),
        (2, 7, 1),
        (3, 6, -1),
    ):
        value = Fraction(sign, 16)
        dual[row][column] = value
        dual[column][row] = value
    return dual


def mutual_kernel_basis() -> Matrix:
    return [
        [1, 0, 0, 0, -1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]


def mutual_kernel_coefficient() -> Matrix:
    return [
        [Fraction(1, 3), 0, 0, 0, Fraction(3, 10), 0, 0, 0, 0, Fraction(1, 6)],
        [0, Fraction(3, 8), 0, Fraction(3, 10), 0, 0, Fraction(-1, 8), 0, 0, 0],
        [0, 0, Fraction(1, 6), 0, 0, 0, 0, 0, 0, 0],
        [0, Fraction(3, 10), 0, Fraction(1, 3), 0, 0, Fraction(-1, 5), 0, 0, 0],
        [Fraction(3, 10), 0, 0, 0, Fraction(3, 8), 0, 0, 0, 0, Fraction(1, 5)],
        [0, 0, 0, 0, 0, Fraction(1, 8), 0, 0, 0, 0],
        [0, Fraction(-1, 8), 0, Fraction(-1, 5), 0, 0, Fraction(3, 8), 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, Fraction(1, 8), 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, Fraction(1, 6), 0],
        [Fraction(1, 6), 0, 0, 0, Fraction(1, 5), 0, 0, 0, 0, Fraction(1, 3)],
    ]


def verify_mutual() -> None:
    tester = build_mutual_tester()
    value = Fraction(3, 4)
    dual_y = mutual_dual_y()
    dual_normalization = matsub(
        smul(value, eye(2)),
        partial_trace_after_first(dual_y, first_dimension=2, rest_dimension=4),
    )
    assert_matrix_equal(
        dual_normalization,
        zeros(2, 2),
        "Mutual QDB dual normalization",
    )
    dual_strategy = matsub(kron(dual_y, eye(2)), tester)
    dual_rank = exact_ldl_psd(dual_strategy, "Mutual QDB dual strategy slack")

    kernel = mutual_kernel_basis()
    assert_matrix_equal(
        matmul(dual_strategy, kernel),
        zeros(16, 10),
        "Mutual QDB dual kernel",
    )
    coefficient = mutual_kernel_coefficient()
    coefficient_rank = exact_ldl_psd(
        coefficient,
        "Mutual QDB kernel coefficient",
    )
    primal = matmul(matmul(kernel, coefficient), transpose(kernel))
    eta = smul(Fraction(1, 2), eye(2))
    assert_matrix_equal(
        partial_trace_last(primal, last_dimension=2),
        kron(eta, eye(4)),
        "Mutual QDB comb normalization",
    )
    primal_value = trace_matmul(tester, primal)
    if primal_value != value:
        raise AssertionError(f"Mutual QDB objective mismatch: {primal_value}")
    if trace_matmul(dual_strategy, primal) != 0:
        raise AssertionError("Mutual QDB complementary slackness failed")

    print("Mutual QDB first-leg exact one-round MF witness")
    print(f"  primal value = dual upper bound = {value}")
    print(f"  dual slack exact LDL rank: {dual_rank}")
    print(f"  kernel coefficient exact LDL rank: {coefficient_rank}")
    print("  Tr_O(S) = eta tensor I_RC exactly, eta=I/2")
    print("  complementary slackness holds exactly")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=("qdb2017", "mutual-entanglement", "all"),
        default="all",
    )
    args = parser.parse_args()

    if args.protocol in {"qdb2017", "all"}:
        verify_qdb2017()
    if args.protocol == "all":
        print()
    if args.protocol in {"mutual-entanglement", "all"}:
        verify_mutual()


if __name__ == "__main__":
    main()
