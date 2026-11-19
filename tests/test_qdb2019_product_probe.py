from __future__ import annotations

import contextlib
import io
import math
import unittest

import numpy as np

import qdb_benchmark_standalone as benchmark
from verification import qdb2019_exact_witness, qdb2019_product_probe

OPTIMUM = (5.0 + math.sqrt(5.0)) / 8.0


def _to_numpy(matrix) -> np.ndarray:
    return np.array([[value.to_float() for value in row] for row in matrix])


def _trace_out_last_qubit(matrix: np.ndarray) -> np.ndarray:
    rest = matrix.shape[0] // 2
    return matrix.reshape(rest, 2, rest, 2).trace(axis1=1, axis2=3)


class FieldTests(unittest.TestCase):
    def test_square_roots(self) -> None:
        sqrt2 = qdb2019_product_probe.SQRT2
        sqrt5 = qdb2019_product_probe.q25(qdb2019_exact_witness.SQRT5)
        self.assertEqual(sqrt2 * sqrt2, qdb2019_product_probe.q25(2))
        self.assertEqual(sqrt5 * sqrt5, qdb2019_product_probe.q25(5))
        self.assertEqual(sqrt2 * sqrt5 * sqrt5 * sqrt2, qdb2019_product_probe.q25(10))
        self.assertAlmostEqual((sqrt2 * sqrt5).to_float(), math.sqrt(10.0), places=14)
        self.assertNotEqual(sqrt2 * sqrt5, sqrt5)
        self.assertTrue((sqrt2 * sqrt5 - sqrt5 * sqrt2).is_zero())


class ProductProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        angle = 3.0 * math.pi / 8.0
        psi = np.array([math.cos(angle), math.sin(angle)])
        self.eta = np.outer(psi, psi)
        tester, _ = benchmark._mf_comb_average_tester("qdb2019")
        self.tester = tester.real

    def test_probe_state(self) -> None:
        np.testing.assert_allclose(
            _to_numpy(qdb2019_product_probe.probe_state()),
            self.eta,
            rtol=0.0,
            atol=1e-15,
        )

    def test_second_step_operator_matches_benchmark(self) -> None:
        # Contract the numerical tester on [Q, R, C, O] with the probe on Q.
        numerical = np.einsum("qipj,pq->ij", self.tester.reshape(2, 8, 2, 8), self.eta)
        exact = qdb2019_product_probe.build_second_step_operator(
            qdb2019_product_probe.probe_state()
        )
        np.testing.assert_allclose(_to_numpy(exact), numerical, rtol=0.0, atol=1e-15)

    def test_eigenvalues(self) -> None:
        operator = _to_numpy(
            qdb2019_product_probe.build_second_step_operator(
                qdb2019_product_probe.probe_state()
            )
        )
        sqrt5 = math.sqrt(5.0)
        expected = sorted(
            2 * [(5.0 + sqrt5) / 32.0, (3.0 + sqrt5) / 32.0]
            + 2 * [(5.0 - sqrt5) / 32.0, (3.0 - sqrt5) / 32.0]
        )
        np.testing.assert_allclose(
            np.linalg.eigvalsh(operator), expected, rtol=0.0, atol=1e-14
        )
        # The exact eigenvalues are listed once each, in decreasing order.
        np.testing.assert_allclose(
            [value.to_float() for value in qdb2019_product_probe.EIGENVALUES],
            expected[::-2],
            rtol=0.0,
            atol=1e-15,
        )

    def test_exact_check(self) -> None:
        result = qdb2019_product_probe.verify()
        self.assertEqual(result["value"], qdb2019_exact_witness.build_dual_y()[0])
        self.assertAlmostEqual(result["value"].to_float(), OPTIMUM, places=14)
        self.assertEqual(result["multiplicities"], [2, 2, 2, 2])

        # The second step is a channel and reaches the optimum for this probe.
        operator = _to_numpy(result["operator"])
        choi = _to_numpy(result["choi"])
        self.assertGreater(np.linalg.eigvalsh(choi).min(), -1e-14)
        np.testing.assert_allclose(
            _trace_out_last_qubit(choi), np.eye(4), rtol=0.0, atol=1e-14
        )
        self.assertAlmostEqual(float(np.trace(operator @ choi)), OPTIMUM, places=14)

        # The comb is a feasible point of the numerical MF SDP of the benchmark.
        comb = _to_numpy(result["comb"])
        self.assertGreater(np.linalg.eigvalsh(comb).min(), -1e-14)
        np.testing.assert_allclose(
            _trace_out_last_qubit(comb),
            np.kron(self.eta, np.eye(4)),
            rtol=0.0,
            atol=1e-14,
        )
        self.assertAlmostEqual(float(np.trace(self.tester @ comb)), OPTIMUM, places=14)

    def test_check_rejects_another_probe(self) -> None:
        # With the probe |0>, W' has other eigenvalues, so the check fails.
        eta = qdb2019_product_probe.lift(qdb2019_exact_witness.projector(0, 0))
        operator = qdb2019_product_probe.build_second_step_operator(eta)
        with self.assertRaises(AssertionError):
            qdb2019_product_probe.spectral_projectors(
                operator, qdb2019_product_probe.EIGENVALUES
            )

    def test_command(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            benchmark.main(["qdb2019-probe"])
        self.assertEqual(
            stdout.getvalue().splitlines()[-1],
            "Conclusion: the probe cos(3 pi/8)|0> + sin(3 pi/8)|1> without "
            "entanglement reaches the optimum (5 + sqrt(5)) / 8.",
        )


if __name__ == "__main__":
    unittest.main()
