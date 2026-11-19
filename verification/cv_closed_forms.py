#!/usr/bin/env python3
"""Closed-form values for the CV-QDB protocol.

These are the numbers of the paper's section "Continuous-variable QDB". The
model uses hbar = 2, so every vacuum quadrature has variance 1. The verifier
prepares a two-mode squeezed vacuum with squeezing r and reads its reference
value from one quadrature of the mode it keeps. A round passes when the value
that the verifier reads from the reply lies within the tolerance delta of the
reference value. The verifier's measurements add no noise, and the honest
prover's reply carries its measured value exactly.

Every pass probability below is the probability that a Gaussian error with
mean zero stays within delta, which is erf(delta / sqrt(2 * variance)). The
module needs only the Python standard library. It calls no solver and reads
no stored result.
"""

from __future__ import annotations

import argparse
import json
import math

# Parameters of the paper: squeezing r, tolerance delta, and the smaller
# amplitude A of the bright probe (the larger amplitude is 2A).
SQUEEZING = 1.2
TOLERANCE = 1.1
PROBE_AMPLITUDE = 10.4

# The adversary reads an amplitude of its probe with a Gaussian error of this
# variance: one unit from the vacuum noise of the coherent probe and one from
# measuring both quadratures of the reply at once (heterodyne detection).
READOUT_ERROR_VARIANCE = 2.0


def pass_probability(error_variance: float, tolerance: float) -> float:
    """Probability that a Gaussian error with mean zero stays within the tolerance."""
    return math.erf(tolerance / math.sqrt(2.0 * error_variance))


def reference_variance(squeezing: float) -> float:
    """Variance cosh(2r) of the verifier's reference value, in either quadrature."""
    return math.cosh(2.0 * squeezing)


def honest_error_variance(squeezing: float) -> float:
    """Variance 2 exp(-2r) of the honest reply minus the reference value."""
    return 2.0 * math.exp(-2.0 * squeezing)


def honest_pass_probability(squeezing: float, tolerance: float) -> float:
    """Honest pass probability of one round, erf(delta exp(r) / 2)."""
    return pass_probability(honest_error_variance(squeezing), tolerance)


def distance_fraud_value(squeezing: float, tolerance: float) -> float:
    """Exact one-round DF value, erf(delta / sqrt(2 cosh(2r))).

    The reply of a distant prover is independent of the reference value, and
    the prover does best when the verifier reads zero. The error is then the
    reference value itself.
    """
    return pass_probability(reference_variance(squeezing), tolerance)


def wrong_quadrature_pass_probability(squeezing: float, tolerance: float) -> float:
    """Pass probability erf(delta / (2 sqrt(cosh(2r)))) with the wrong quadrature.

    An adversary that measures the other quadrature of the verifier's
    challenge mode returns a value that is independent of the reference value
    and has the same variance, so the two variances add.
    """
    return pass_probability(2.0 * reference_variance(squeezing), tolerance)


def amplitude_mixup_probability(amplitude: float) -> float:
    """Probability (1 - erf(A / 4)) / 2 of taking one probe amplitude for the other.

    The amplitudes are A and 2A. The adversary mistakes one for the other when
    its read-out error exceeds half the gap between them, A / 2, in the
    direction of the other amplitude.
    """
    half_gap = 0.5 * amplitude
    return 0.5 * math.erfc(half_gap / math.sqrt(2.0 * READOUT_ERROR_VARIANCE))


def attack_success_probability(
    squeezing: float, tolerance: float, amplitude: float
) -> float:
    """Success probability of the bright-probe MF attack with amplitudes A and 2A.

    The adversary reads from the prover's reply which quadrature the prover
    measured and which quadrature it wrote into. The only error counted is the
    mix-up of the two amplitudes, which gives the wrong measured quadrature.
    Without this error the adversary passes like the honest prover, and with
    it the adversary measures the wrong quadrature of the challenge mode.
    """
    mixup = amplitude_mixup_probability(amplitude)
    return (1.0 - mixup) * honest_pass_probability(
        squeezing, tolerance
    ) + mixup * wrong_quadrature_pass_probability(squeezing, tolerance)


def mafia_fraud_table_entry(squeezing: float, tolerance: float) -> float:
    """Honest pass probability rounded down to four decimals.

    Table 1 of the paper lists the CV MF value as at least this number. The
    bright-probe attack approaches the honest pass probability as the probe
    gets brighter, so a bright enough probe reaches this number.
    """
    return math.floor(honest_pass_probability(squeezing, tolerance) * 1e4) / 1e4


def probe_mean_photon_number(amplitude: float) -> float:
    """Mean photon number 5 A^2 / 4 of a coherent probe with amplitudes A and 2A."""
    return 5.0 * amplitude**2 / 4.0


def challenge_mean_photon_number(squeezing: float) -> float:
    """Mean photon number sinh(r)^2 of the verifier's challenge mode."""
    return math.sinh(squeezing) ** 2


def compute_values(
    squeezing: float = SQUEEZING,
    tolerance: float = TOLERANCE,
    amplitude: float = PROBE_AMPLITUDE,
) -> dict[str, float]:
    """Return the CV-QDB numbers of the paper for the given parameters."""
    return {
        "squeezing": squeezing,
        "tolerance": tolerance,
        "honest_error_variance": honest_error_variance(squeezing),
        "honest_pass_probability": honest_pass_probability(squeezing, tolerance),
        "reference_variance": reference_variance(squeezing),
        "distance_fraud_value": distance_fraud_value(squeezing, tolerance),
        "probe_amplitude_small": amplitude,
        "probe_amplitude_large": 2.0 * amplitude,
        "readout_error_variance": READOUT_ERROR_VARIANCE,
        "amplitude_mixup_probability": amplitude_mixup_probability(amplitude),
        "wrong_quadrature_pass_probability": wrong_quadrature_pass_probability(
            squeezing, tolerance
        ),
        "attack_success_probability": attack_success_probability(
            squeezing, tolerance, amplitude
        ),
        "mafia_fraud_table_entry": mafia_fraud_table_entry(squeezing, tolerance),
        "probe_mean_photon_number": probe_mean_photon_number(amplitude),
        "challenge_mean_photon_number": challenge_mean_photon_number(squeezing),
    }


def _print_values(values: dict[str, float]) -> None:
    def row(label: str, text: str) -> None:
        print(f"  {label:<55}{text}")

    print("CV-QDB closed-form values (hbar = 2, vacuum variance 1)")
    row("squeezing r", f"{values['squeezing']:g}")
    row("tolerance delta", f"{values['tolerance']:g}")
    print()
    print("Honest prover")
    row("error variance, 2 exp(-2r)", f"{values['honest_error_variance']:.6f}")
    row(
        "pass probability, erf(delta exp(r) / 2)",
        f"{values['honest_pass_probability']:.6f}",
    )
    print()
    print("Distance fraud")
    row("reference variance, cosh(2r)", f"{values['reference_variance']:.6f}")
    row(
        "exact DF value, erf(delta / sqrt(2 cosh(2r)))",
        f"{values['distance_fraud_value']:.6f}",
    )
    print()
    print("Mafia fraud, bright-probe attack")
    row(
        "probe amplitudes A and 2A",
        f"{values['probe_amplitude_small']:g} and {values['probe_amplitude_large']:g}",
    )
    row("read-out error variance", f"{values['readout_error_variance']:g}")
    row(
        "amplitude mix-up, (1 - erf(A / 4)) / 2",
        f"{values['amplitude_mixup_probability']:.3e}",
    )
    row(
        "wrong quadrature, erf(delta / (2 sqrt(cosh(2r))))",
        f"{values['wrong_quadrature_pass_probability']:.6f}",
    )
    row("attack success probability", f"{values['attack_success_probability']:.6f}")
    row(
        "CV MF entry of Table 1 (honest pass, rounded down)",
        f">= {values['mafia_fraud_table_entry']:.4f}",
    )
    row(
        "mean photon number of the probe, 5 A^2 / 4",
        f"{values['probe_mean_photon_number']:.1f}",
    )
    row(
        "mean photon number of the challenge mode, sinh(r)^2",
        f"{values['challenge_mean_photon_number']:.3f}",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print the closed-form CV-QDB values of the paper: the honest pass "
            "probability, the exact distance-fraud value, and the bright-probe "
            "mafia-fraud attack."
        )
    )
    parser.add_argument(
        "--squeezing",
        type=float,
        default=SQUEEZING,
        help=f"Squeezing r of the verifier's EPR pair. Default: {SQUEEZING}.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=TOLERANCE,
        help=f"Acceptance tolerance delta. Default: {TOLERANCE}.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=PROBE_AMPLITUDE,
        help=(
            "Smaller amplitude A of the bright probe; the larger one is 2A. "
            f"Default: {PROBE_AMPLITUDE}."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON only.",
    )
    args = parser.parse_args(argv)
    if args.squeezing < 0.0 or args.tolerance <= 0.0 or args.amplitude <= 0.0:
        parser.error(
            "--squeezing must be nonnegative; --tolerance and --amplitude "
            "must be positive"
        )

    values = compute_values(args.squeezing, args.tolerance, args.amplitude)
    if args.json:
        print(json.dumps(values, indent=2))
        return
    _print_values(values)


if __name__ == "__main__":
    main()
