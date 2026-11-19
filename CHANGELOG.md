# Changelog

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
