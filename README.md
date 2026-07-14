# qdb-benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21307978.svg)](https://doi.org/10.5281/zenodo.21307978)

Code and data artifact for the paper

> **Security evaluation of quantum distance-bounding protocols via
> semidefinite programming**
> Kevin Bogner, Aysajan Abidin, Dave Singelée, Bart Preneel

The artifact builds the paper's reduced one-round fast-phase games as
semidefinite programs (SDPs), solves the distance-fraud and mafia-fraud
programs for the discrete-variable QDB protocols, runs the restricted
Gaussian finite-grid estimator for the CV-QDB protocol, and checks the
discrete-variable mafia-fraud values with solver-free exact-arithmetic
primal and dual certificates. The shipped NumPy files retain the raw
numerical solver witnesses for reproducibility.

## Installation

Requires Python 3.10 or newer. With [uv](https://docs.astral.sh/uv/):

```
uv sync
```

This creates a virtual environment from the pinned `uv.lock`. The
environment is large (under 1 GB, mainly PyTorch). Plain
`pip install -e .` works as well but does not use the pinned versions.

Protocol keys used by all commands: `qdb2019` (QDB 2019),
`mutual-entanglement` (Mutual QDB), `e91` (E91 QDB), `cv` (CV-QDB), and
`qdb2017` (QDB 2017, treated in the paper's Appendix B).

## Reproducing the paper's values

**Benchmark table (Table 1).**

```
uv run qdb-benchmark table
```

Solves the DF and MF programs for all protocols and prints the values,
together with the honest-run completeness of each protocol, as JSON.
The DV values are the Table 1 entries (and the QDB 2017 values of
Appendix B); the CV entries are the Gaussian benchmark and the
finite-grid MF estimate. Takes about a minute and a half on a laptop;
the CV grid dominates.

**Exact discrete-variable mafia-fraud certificates (Tables 1, 3, and 4).**

```
uv run qdb-benchmark exact-certificate --protocol all
```

The verifier independently rebuilds every DV tester and checks matching
primal and dual witnesses using exact arithmetic only. It does not call
an SDP solver or rely on the floating-point `.npz` files. It proves

```
QDB 2017 rapid phase                    1
QDB 2019                               (5 + sqrt(5)) / 8
Mutual QDB first timed return leg       3 / 4
E91 QDB retained value-check branches  (8 + sqrt(24 + 17 sqrt(2))) / 16
```

The rational cases use `fractions.Fraction`; QDB 2019 is checked over
`Q(sqrt(5))`; and E91 is checked over
`Q(sqrt(2), sqrt(24 + 17 sqrt(2)))`. Positivity is established by exact
LDL decompositions. The E91 verifier decides algebraic signs using
rational square-root enclosures.

**Numerical solver witnesses.**

```
uv run qdb-benchmark certificate --protocol qdb2019 --check-only
```

(similarly for the other DV protocol keys). This NumPy command rebuilds
the tester and reports the residuals, eigenvalues, and duality gap of the
stored solver output. Its floating-point repair and candidate bracket
are diagnostics, not a mathematical certificate; use
`exact-certificate` for the proof.

**CV calibration sweep (Figure 2).**

```
uv run qdb-benchmark cv-sweep
```

Writes `results/cv_calibration_sweep.{csv,json}` and
`figures/cv_calibration_sweep.{png,pdf}`. The default paper grid takes
about three minutes on a laptop; `--grid-profile quick` runs a coarser
exploratory grid.

**Single protocol.**

```
uv run qdb-benchmark sdp --protocol e91
```

Solves one protocol's programs (`--solver SCS` or `--solver CLARABEL`;
CV grid parameters are exposed as `--cv-*` flags, see `--help`).

**Regenerating the numerical solver witnesses.**

```
uv run qdb-benchmark certificate --protocol qdb2019
```

Re-solves the primal and dual MF comb SDPs, saves fresh numerical
witnesses to `certificates/`, and checks their floating-point residuals.

## Layout

- `qdb_benchmark_standalone.py` — numerical protocol models, SDP
  construction, benchmark routines, and the CLI.
- `verification/` — standalone exact primal/dual certificate verifiers
  and the combined `exact-certificate` entry point.
- `tests/` — exact-tester regression checks against the numerical
  protocol implementation and end-to-end witness checks.
- `certificates/` — shipped numerical MF solver witnesses (`.npz` files
  storing the primal comb, probe state, dual point, and protocol metadata).
- `results/`, `figures/` — output of the CV calibration sweep as used
  in the paper.

Run the regression suite with:

```
uv run python -m unittest discover -v
```

## License

MIT, see `LICENSE`.

## Citing

A citation entry will be added once the paper is public.
