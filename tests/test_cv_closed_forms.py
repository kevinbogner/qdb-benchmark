from __future__ import annotations

import contextlib
import io
import json
import math
import unittest

import qdb_benchmark_standalone as benchmark
from verification import cv_closed_forms


class PaperValueTests(unittest.TestCase):
    """The numbers of the paper's section "Continuous-variable QDB"."""

    def setUp(self) -> None:
        self.values = cv_closed_forms.compute_values()

    def assert_prints_as(self, key: str, format_spec: str, expected: str) -> None:
        self.assertEqual(format(self.values[key], format_spec), expected)

    def test_parameters(self) -> None:
        self.assertEqual(self.values["squeezing"], 1.2)
        self.assertEqual(self.values["tolerance"], 1.1)
        self.assertEqual(self.values["probe_amplitude_small"], 10.4)
        self.assertEqual(self.values["probe_amplitude_large"], 20.8)
        self.assertEqual(self.values["readout_error_variance"], 2.0)

    def test_honest_prover(self) -> None:
        self.assert_prints_as("honest_error_variance", ".2f", "0.18")
        self.assert_prints_as("honest_pass_probability", ".5f", "0.99019")
        self.assertAlmostEqual(
            self.values["honest_pass_probability"],
            math.erf(1.1 * math.exp(1.2) / 2.0),
            places=14,
        )

    def test_distance_fraud(self) -> None:
        self.assert_prints_as("reference_variance", ".2f", "5.56")
        self.assert_prints_as("distance_fraud_value", ".4f", "0.3592")
        self.assertAlmostEqual(
            self.values["distance_fraud_value"],
            math.erf(1.1 / math.sqrt(2.0 * math.cosh(2.4))),
            places=14,
        )

    def test_bright_probe_attack(self) -> None:
        self.assert_prints_as("amplitude_mixup_probability", ".2e", "1.18e-04")
        self.assertAlmostEqual(
            self.values["amplitude_mixup_probability"],
            (1.0 - math.erf(10.4 / 4.0)) / 2.0,
            places=14,
        )
        self.assert_prints_as("wrong_quadrature_pass_probability", ".4f", "0.2586")
        self.assertAlmostEqual(
            self.values["wrong_quadrature_pass_probability"],
            math.erf(1.1 / (2.0 * math.sqrt(math.cosh(2.4)))),
            places=14,
        )
        self.assert_prints_as("attack_success_probability", ".6f", "0.990103")

    def test_table_entry(self) -> None:
        # Table 1 lists the CV MF value as at least 0.9901, the honest pass
        # probability rounded down. The attack with amplitudes 10.4 and 20.8
        # reaches this entry and stays below the honest pass probability.
        self.assertEqual(self.values["mafia_fraud_table_entry"], 0.9901)
        self.assertGreaterEqual(self.values["attack_success_probability"], 0.9901)
        self.assertLess(
            self.values["attack_success_probability"],
            self.values["honest_pass_probability"],
        )

    def test_photon_numbers(self) -> None:
        self.assert_prints_as("probe_mean_photon_number", ".0f", "135")
        self.assert_prints_as("challenge_mean_photon_number", ".2f", "2.28")
        self.assert_prints_as("challenge_mean_photon_number", ".1f", "2.3")


class GaussianModelTests(unittest.TestCase):
    """The closed forms agree with the Gaussian protocol model of the benchmark."""

    def setUp(self) -> None:
        self.protocol = benchmark.CVQDB()
        self.values = cv_closed_forms.compute_values(
            squeezing=self.protocol.epr_r,
            tolerance=float(self.protocol.accept_tol),
        )

    def test_reference_variance(self) -> None:
        for a in (0, 1):
            self.assertAlmostEqual(
                float(self.protocol.var_m(a)),
                self.values["reference_variance"],
                places=12,
            )

    def test_honest_error_variance(self) -> None:
        for a in (0, 1):
            gain = float(self.protocol.honest_response_gain(a, 0))
            error_variance = (
                float(self.protocol.var_m(a))
                + gain * gain * float(self.protocol.var_m_prime(a))
                - 2.0 * gain * float(self.protocol.cov_m_m_prime(a))
            )
            self.assertAlmostEqual(
                error_variance,
                self.values["honest_error_variance"],
                places=12,
            )

    def test_honest_pass_probability(self) -> None:
        self.assertAlmostEqual(
            benchmark._cv_protocol_completeness(self.protocol),
            self.values["honest_pass_probability"],
            places=9,
        )

    def test_other_quadrature_is_uncorrelated(self) -> None:
        # Covariance ordering: (x_A, x_B, p_A, p_B). The verifier's x and the
        # prover-side p are uncorrelated and have the same variance, so a reply
        # built from the wrong quadrature misses the reference value by an
        # error of twice the reference variance.
        cov = self.protocol.cov
        self.assertAlmostEqual(float(cov[0, 3]), 0.0, places=12)
        self.assertAlmostEqual(float(cov[2, 1]), 0.0, places=12)
        for index in (1, 3):
            self.assertAlmostEqual(
                float(cov[index, index]),
                self.values["reference_variance"],
                places=12,
            )


class CommandTests(unittest.TestCase):
    def test_cv_values_command(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            benchmark.main(["cv-values", "--json"])
        self.assertEqual(
            json.loads(stdout.getvalue()),
            cv_closed_forms.compute_values(),
        )

    def test_table_row(self) -> None:
        row = benchmark._cv_table_entry()
        values = cv_closed_forms.compute_values()
        self.assertEqual(row["protocol"], "cv")
        self.assertEqual(row["df_status"], "closed_form")
        self.assertEqual(row["df_value"], values["distance_fraud_value"])
        self.assertEqual(row["mf_status"], "attack_lower_bound")
        self.assertEqual(row["mf_value"], values["attack_success_probability"])
        self.assertEqual(row["completeness_status"], "closed_form")
        self.assertEqual(row["completeness_value"], values["honest_pass_probability"])
        self.assertEqual(row["cv_values"], values)


if __name__ == "__main__":
    unittest.main()
