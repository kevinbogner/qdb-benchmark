# Changelog

## 1.0.3

This release matches the revised version of the paper. Release 1.0.2, with
the version DOI 10.5281/zenodo.24375842, matches an earlier state of the
revision. Two later changes of the paper concern the artifact, and this
release follows both. No computed value changes, and the exact certificate
verifier and its output are unchanged.

- The paper now describes in words how bright the probe of the bright-probe
  attack on CV-QDB must be, and it no longer prints the numbers of an
  example with amplitudes 10.4 and 20.8 (mix-up probability 1.18e-4,
  success probability 0.990103, 135 probe photons on average against 2.28
  in the challenge mode). `qdb-benchmark cv-values` still evaluates these
  numbers. It now presents them as an example in the paper's model, in
  which the prover's reply carries its measured value exactly, and says
  that the paper does not print them. The honest pass probability 0.99019,
  the exact distance-fraud value 0.3592, and the Table 1 entry of at least
  0.9901 remain the values of the paper.
- In the JSON output of `cv-values`, and in the CV row of
  `qdb-benchmark table`, the numbers of the example moved under the key
  `bright_probe_example`. The mafia-fraud value of the CV row is now the
  Table 1 entry 0.9901, where it was the success probability of the
  example.
- Added `verification/qdb2019_product_probe.py` and the command
  `qdb-benchmark qdb2019-probe`. The paper now says that the probe of the
  optimal attack on QDB 2019 needs no entanglement with the adversary's
  memory and that `cos(3 pi/8)|0> + sin(3 pi/8)|1>` is an optimal probe.
  The witness of `exact-certificate` uses a probe with the reduced state
  `I/2`, so it does not show this statement. The new command proves it in
  exact arithmetic over `Q(sqrt(2), sqrt(5))`, and
  `tests/test_qdb2019_product_probe.py` checks the new module.
- The README describes both changes and names the releases that match each
  version of the paper.

## 1.0.2

This release matches the revised version of the paper. The revised paper
no longer estimates the CV-QDB mafia-fraud value on a finite grid. The
published protocol does not bound the energy of the probe, so the paper
analyses an explicit bright-probe attack instead and lists the value as
at least 0.9901. It also has no calibration figure. No discrete-variable
value changes, and the exact certificate verifier is unchanged.

- Added `verification/cv_closed_forms.py` and the command
  `qdb-benchmark cv-values`. They evaluate the CV-QDB numbers of the
  revised paper: the honest pass probability 0.99019, the exact
  distance-fraud value 0.3592, and the bright-probe attack with amplitudes
  10.4 and 20.8 (mix-up probability 1.18e-4, success probability 0.990103,
  135 probe photons on average). `tests/test_cv_closed_forms.py` checks
  these numbers.
- The CV row of `qdb-benchmark table` now holds these closed-form values.
  Its mafia-fraud entry is the success probability of the attack, a lower
  bound, where it was the grid estimate 0.9138.
- Removed `results/cv_calibration_sweep.csv`,
  `results/cv_calibration_sweep.json`, and
  `figures/cv_calibration_sweep.png`, the data and the image of the
  calibration figure. They remain in release 1.0.1.
- Kept the finite-grid estimator (`sdp --protocol cv` and `cv-sweep`) for
  readers of the paper's first version. Its output and its documentation
  now say that the revised paper does not use it.
- The `e91` values are now described as values of the fixed E91 QDB
  protocol of the revised paper, in which the prover does not announce its
  measurement settings. The honest-session summary of `e91` therefore
  reports `requires_prover_setting_disclosure` as false. The game and its
  values are unchanged.
- The README follows the revised paper, which has no Table 4, no
  appendices, and no calibration figure, and it cites the paper
  (arXiv:2607.13464).
- `__version__` in `qdb_benchmark_standalone.py` now carries the release
  number.

## 1.0.1

This release matches the first version of the paper, arXiv:2607.13464v1.
Its version DOI is 10.5281/zenodo.21361445.

- Added exact-arithmetic primal and dual certificates for the four
  discrete-variable mafia-fraud values, the verifier
  `qdb-benchmark exact-certificate`, and regression tests. The stored
  NumPy witnesses became numerical diagnostics.

## 1.0

First release.
