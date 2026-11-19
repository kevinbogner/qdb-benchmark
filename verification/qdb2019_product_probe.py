#!/usr/bin/env python3
"""Exact check of an optimal QDB 2019 probe without entanglement.

The paper's section "QDB 2019" says that the probe of the optimal one-round MF
attack needs no entanglement with the adversary's memory. An optimal probe is
the state

    |psi> = cos(3 pi/8)|0> + sin(3 pi/8)|1>,

and the adversary stores the prover's reply and applies a joint operation to
the reply and the challenge. This module proves the statement with exact
arithmetic in Q(sqrt(2), sqrt(5)). No numerical SDP solver is called.

The prover measures the probe in basis a and prepares its outcome t in basis
b, so its reply is the state sum_t |<t|_a psi>|^2 |t>_b<t|. The second step of
the adversary is a channel from the reply R and the challenge C to the output
O. A channel with Choi matrix J succeeds with probability Tr(W' J), where W'
is the average over the branches (a, b, c) of

    (reply tensor |c>_a<c|)^T tensor |c>_b<c|.

The module checks three facts.

1. W' has the eigenvalues (5 +- sqrt(5))/32 and (3 +- sqrt(5))/32, each with
   multiplicity two.
2. Twice the projector P onto the eigenspace of the largest eigenvalue is the
   Choi matrix of a channel. That is, J = 2P is positive semidefinite and
   Tr_O(J) = I_RC.
3. This channel succeeds with probability Tr(W' J) = (5 + sqrt(5))/8.

The Choi matrix of every channel from R and C to O is positive semidefinite
and has trace four, so no second step succeeds with a probability above four
times the largest eigenvalue. The channel of fact 3 reaches this bound. Its
value (5 + sqrt(5))/8 is also the optimum over all attacks, which
``qdb2019_exact_witness.py`` proves, so the probe is optimal. The module takes
the tester of the full game from ``qdb2019_exact_witness.py`` and confirms
that the comb |psi><psi| tensor J is a feasible point of the MF SDP with the
optimal value.

Run it from the root of the artifact as
``python3 -m verification.qdb2019_product_probe``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction

from .qdb2019_exact_witness import (
    Q5,
    assert_matrix_equal,
    build_dual_y,
    build_qdb2019_tester,
    matadd,
    matsub,
    projector,
    q,
    q5,
    smul,
    transpose,
)


@dataclass(frozen=True)
class Q25:
    """Element x + y sqrt(2) of Q(sqrt(2), sqrt(5)), with x, y in Q(sqrt(5))."""

    x: Q5 = Q5()
    y: Q5 = Q5()

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", q5(self.x))
        object.__setattr__(self, "y", q5(self.y))

    def __add__(self, other: object) -> Q25:
        other = q25(other)
        return Q25(self.x + other.x, self.y + other.y)

    __radd__ = __add__

    def __neg__(self) -> Q25:
        return Q25(-self.x, -self.y)

    def __sub__(self, other: object) -> Q25:
        return self + (-q25(other))

    def __rsub__(self, other: object) -> Q25:
        return q25(other) - self

    def __mul__(self, other: object) -> Q25:
        other = q25(other)
        return Q25(
            self.x * other.x + 2 * self.y * other.y,
            self.x * other.y + self.y * other.x,
        )

    __rmul__ = __mul__

    def is_zero(self) -> bool:
        return self.x.is_zero() and self.y.is_zero()

    def to_float(self) -> float:
        return self.x.to_float() + self.y.to_float() * math.sqrt(2.0)

    def __str__(self) -> str:
        if self.y.is_zero():
            return str(self.x)
        if self.x.is_zero():
            return f"({self.y})*sqrt2"
        return f"{self.x} + ({self.y})*sqrt2"


def q25(value: object) -> Q25:
    """Embed a rational number or an element of Q(sqrt(5))."""
    if isinstance(value, Q25):
        return value
    return Q25(q5(value))


ZERO = Q25()
ONE = q25(1)
SQRT2 = Q25(Q5(), Q5(1))

Matrix = list[list[Q25]]

# Eigenvalues of W' in decreasing order, as elements of Q(sqrt(5)). Each has
# the multiplicity below.
EIGENVALUES = (
    q(Fraction(5, 32), Fraction(1, 32)),
    q(Fraction(3, 32), Fraction(1, 32)),
    q(Fraction(5, 32), Fraction(-1, 32)),
    q(Fraction(3, 32), Fraction(-1, 32)),
)
MULTIPLICITY = 2


def lift(matrix: list[list[Q5]]) -> Matrix:
    """Embed a matrix over Q(sqrt(5)) in Q(sqrt(2), sqrt(5))."""
    return [[q25(value) for value in row] for row in matrix]


def zeros(rows: int, columns: int) -> Matrix:
    return [[ZERO for _ in range(columns)] for _ in range(rows)]


def eye(size: int) -> Matrix:
    return [[ONE if i == j else ZERO for j in range(size)] for i in range(size)]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    return [
        [
            sum((left[i][k] * right[k][j] for k in range(len(right))), ZERO)
            for j in range(len(right[0]))
        ]
        for i in range(len(left))
    ]


def kron(left: Matrix, right: Matrix) -> Matrix:
    return [
        [
            left_value * right_value
            for left_value in left_row
            for right_value in right_row
        ]
        for left_row in left
        for right_row in right
    ]


def trace(matrix: Matrix) -> Q25:
    return sum((matrix[i][i] for i in range(len(matrix))), ZERO)


def trace_matmul(left: Matrix, right: Matrix) -> Q25:
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
    return [
        [
            sum(
                (
                    matrix[i * last_dimension + k][j * last_dimension + k]
                    for k in range(last_dimension)
                ),
                ZERO,
            )
            for j in range(rest_dimension)
        ]
        for i in range(rest_dimension)
    ]


def probe_state() -> Matrix:
    """Return |psi><psi| for |psi> = cos(3 pi/8)|0> + sin(3 pi/8)|1>.

    The double-angle formulas with cos(3 pi/4) = -sqrt(2)/2 and
    sin(3 pi/4) = sqrt(2)/2 give the entries cos(3 pi/8)^2 = (2 - sqrt(2))/4,
    sin(3 pi/8)^2 = (2 + sqrt(2))/4, and
    cos(3 pi/8) sin(3 pi/8) = sqrt(2)/4.
    """
    half = q25(Fraction(1, 2))
    off_diagonal = q25(Fraction(1, 4)) * SQRT2
    return [
        [half - off_diagonal, off_diagonal],
        [off_diagonal, half + off_diagonal],
    ]


def reply_state(eta: Matrix, a: int, b: int) -> Matrix:
    """Return the prover's reply to the probe state eta on the branch (a, b).

    The prover measures the probe in basis a and prepares its outcome t in
    basis b, so the reply is sum_t <t|_a eta |t>_a |t>_b<t|.
    """
    reply = zeros(2, 2)
    for t in (0, 1):
        probability = trace_matmul(lift(projector(a, t)), eta)
        reply = matadd(reply, smul(probability, lift(projector(b, t))))
    return reply


def build_second_step_operator(eta: Matrix) -> Matrix:
    """Return W' for the probe state eta, in the register order [R, C, O].

    W' is the average over the branches (a, b, c) of
    (reply tensor |c>_a<c|)^T tensor |c>_b<c|.
    """
    operator = zeros(8, 8)
    for a in (0, 1):
        for b in (0, 1):
            reply = reply_state(eta, a, b)
            for c in (0, 1):
                challenge = lift(projector(a, c))
                accept = lift(projector(b, c))
                operator = matadd(
                    operator,
                    kron(transpose(kron(reply, challenge)), accept),
                )
    return smul(q25(Fraction(1, 8)), operator)


def contract_probe(tester: Matrix, eta: Matrix) -> Matrix:
    """Return Tr_Q[tester (eta tensor I_RCO)] for a tester on [Q, R, C, O]."""
    rest_dimension = len(tester) // 2
    return [
        [
            sum(
                (
                    tester[q_row * rest_dimension + i][q_column * rest_dimension + j]
                    * eta[q_column][q_row]
                    for q_row in (0, 1)
                    for q_column in (0, 1)
                ),
                ZERO,
            )
            for j in range(rest_dimension)
        ]
        for i in range(rest_dimension)
    ]


def spectral_projectors(operator: Matrix, eigenvalues: tuple[Q5, ...]) -> list[Matrix]:
    """Return the projectors onto the eigenspaces of a symmetric matrix.

    For distinct numbers lambda_i, the candidate P_i is the product of
    (operator - lambda_j I) / (lambda_i - lambda_j) over all j != i. The
    function checks exactly that every P_i is a symmetric projector with
    operator P_i = lambda_i P_i and that the P_i sum to the identity. Then
    operator = sum_i lambda_i P_i, so the lambda_i are the eigenvalues of the
    operator, and lambda_i has multiplicity Tr(P_i).
    """
    assert_matrix_equal(operator, transpose(operator), "operator symmetry")
    identity = eye(len(operator))
    projectors: list[Matrix] = []
    for i, eigenvalue in enumerate(eigenvalues):
        candidate = identity
        for j, other in enumerate(eigenvalues):
            if j == i:
                continue
            factor = matsub(operator, smul(q25(other), identity))
            scale = q25(1 / (eigenvalue - other))
            candidate = matmul(candidate, smul(scale, factor))
        assert_matrix_equal(candidate, transpose(candidate), f"P_{i} symmetry")
        assert_matrix_equal(matmul(candidate, candidate), candidate, f"P_{i} squared")
        assert_matrix_equal(
            matmul(operator, candidate),
            smul(q25(eigenvalue), candidate),
            f"P_{i} eigenspace",
        )
        projectors.append(candidate)

    total = zeros(len(operator), len(operator))
    for candidate in projectors:
        total = matadd(total, candidate)
    assert_matrix_equal(total, identity, "sum of the eigenprojectors")
    return projectors


def verify() -> dict[str, object]:
    """Run all exact checks and return the verified objects."""
    eta = probe_state()
    assert_matrix_equal(eta, transpose(eta), "probe symmetry")
    assert_matrix_equal(matmul(eta, eta), eta, "probe purity")
    if trace(eta) != ONE:
        raise AssertionError(f"probe trace is not one: {trace(eta)}")

    # W' from the game, and again from the tester of the exact MF witness.
    operator = build_second_step_operator(eta)
    tester = lift(build_qdb2019_tester())
    assert_matrix_equal(
        operator,
        contract_probe(tester, eta),
        "second-step operator against the tester",
    )

    # Fact 1: the eigenvalues of W' and their multiplicities.
    for larger, smaller in zip(EIGENVALUES, EIGENVALUES[1:]):
        if (larger - smaller).sign() <= 0:
            raise AssertionError("eigenvalues are not in decreasing order")
    projectors = spectral_projectors(operator, EIGENVALUES)
    for i, candidate in enumerate(projectors):
        if trace(candidate) != q25(MULTIPLICITY):
            raise AssertionError(f"multiplicity of eigenvalue {i}: {trace(candidate)}")

    # Fact 2: J = 2P is the Choi matrix of a channel. It is positive
    # semidefinite because P is a symmetric projector.
    choi = smul(q25(2), projectors[0])
    assert_matrix_equal(partial_trace_last(choi, 2), eye(4), "channel normalization")

    # Fact 3: the value of this channel is the optimum of the exact MF witness.
    optimum, _ = build_dual_y()
    value = trace_matmul(operator, choi)
    if value != q25(optimum):
        raise AssertionError(f"objective mismatch: {value} != {optimum}")
    if 4 * EIGENVALUES[0] != optimum:
        raise AssertionError("four times the largest eigenvalue is not the optimum")

    # The comb of the full game. It is positive semidefinite because eta and P
    # are symmetric projectors.
    comb = kron(eta, choi)
    assert_matrix_equal(
        partial_trace_last(comb, 2),
        kron(eta, eye(4)),
        "comb normalization",
    )
    comb_value = trace_matmul(tester, comb)
    if comb_value != q25(optimum):
        raise AssertionError(f"comb objective mismatch: {comb_value} != {optimum}")

    return {
        "eta": eta,
        "operator": operator,
        "eigenvalues": EIGENVALUES,
        "multiplicities": [MULTIPLICITY] * len(EIGENVALUES),
        "choi": choi,
        "comb": comb,
        "value": optimum,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prove in exact arithmetic that the probe "
            "cos(3 pi/8)|0> + sin(3 pi/8)|1>, without entanglement with the "
            "adversary's memory, reaches the optimal one-round MF value of "
            "QDB 2019."
        )
    )
    parser.parse_args(argv)

    result = verify()
    value = result["value"]

    print("QDB 2019 exact check of an optimal probe without entanglement")
    print("probe: |psi> = cos(3 pi/8)|0> + sin(3 pi/8)|1>")
    print("the probe is a pure state, so it is not entangled with a memory")
    print("register order: R,C,O")
    print("number field: Q(sqrt(2), sqrt(5))")
    print()
    print("Second-step operator W'")
    print("  W' = Tr_Q[Wbar (|psi><psi| tensor I_RCO)] exactly, for the tester Wbar")
    print("  W' = sum_i lambda_i P_i with symmetric projectors P_i that sum to I")
    for i, (eigenvalue, multiplicity) in enumerate(
        zip(result["eigenvalues"], result["multiplicities"])
    ):
        print(
            f"    lambda_{i} = {eigenvalue} ~= {eigenvalue.to_float():.12f}, "
            f"multiplicity Tr(P_{i}) = {multiplicity}"
        )
    print()
    print("Second step J = 2 P_0")
    print("  J is positive semidefinite, because P_0 is a symmetric projector")
    print("  Tr_O(J) = I_RC exactly, so J is the Choi matrix of a channel")
    print(f"  Tr(W' J) = {value} ~= {value.to_float():.15f}")
    print("  Tr(W' J) = 4 lambda_0, so no channel does better with this probe")
    print()
    print("Comb S = |psi><psi| tensor J of the full game")
    print("  Tr_O(S) = |psi><psi| tensor I_RC exactly")
    print(f"  Tr(Wbar S) = {value} ~= {value.to_float():.15f}")
    print("  Tr(Wbar S) equals the optimum mu of the exact QDB 2019 MF witness")
    print()
    print(
        "Conclusion: the probe cos(3 pi/8)|0> + sin(3 pi/8)|1> without "
        "entanglement reaches the optimum (5 + sqrt(5)) / 8."
    )


if __name__ == "__main__":
    main()
