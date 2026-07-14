from __future__ import annotations

import unittest

import numpy as np

import qdb_benchmark_standalone as benchmark
from verification import (
    e91_exact_witness,
    qdb2019_exact_witness,
    rational_exact_witnesses,
)


def _rational_to_numpy(matrix) -> np.ndarray:
    return np.array([[float(value) for value in row] for row in matrix])


def _q5_to_numpy(matrix) -> np.ndarray:
    return np.array([[value.to_float() for value in row] for row in matrix])


def _q4_to_numpy(matrix) -> np.ndarray:
    return np.array([[value.to_float() for value in row] for row in matrix])


class ExactTesterTests(unittest.TestCase):
    def assert_matches_benchmark(
        self,
        protocol: str,
        exact: np.ndarray,
    ) -> None:
        numerical, _ = benchmark._mf_comb_average_tester(protocol)
        np.testing.assert_allclose(
            exact,
            numerical.real,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            numerical.imag,
            0.0,
            rtol=0.0,
            atol=1e-15,
        )

    def test_qdb2017_tester(self) -> None:
        self.assert_matches_benchmark(
            "qdb2017",
            _rational_to_numpy(rational_exact_witnesses.build_qdb2017_tester()),
        )

    def test_qdb2019_tester(self) -> None:
        self.assert_matches_benchmark(
            "qdb2019",
            _q5_to_numpy(qdb2019_exact_witness.build_qdb2019_tester()),
        )

    def test_mutual_tester(self) -> None:
        self.assert_matches_benchmark(
            "mutual-entanglement",
            _rational_to_numpy(rational_exact_witnesses.build_mutual_tester()),
        )

    def test_e91_tester(self) -> None:
        self.assert_matches_benchmark(
            "e91",
            _q4_to_numpy(e91_exact_witness.build_e91_tester()),
        )


class ExactWitnessTests(unittest.TestCase):
    def test_qdb2017_witness(self) -> None:
        rational_exact_witnesses.verify_qdb2017()

    def test_qdb2019_witness(self) -> None:
        qdb2019_exact_witness.main()

    def test_mutual_witness(self) -> None:
        rational_exact_witnesses.verify_mutual()

    def test_e91_witness(self) -> None:
        e91_exact_witness.main()


if __name__ == "__main__":
    unittest.main()
