# qdb-benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21307978.svg)](https://doi.org/10.5281/zenodo.21307978)

Code and data artifact for the paper

> **Security evaluation of quantum distance-bounding protocols via
> semidefinite programming**
> Kevin Bogner, Aysajan Abidin, Dave Singelée, Bart Preneel
> [arXiv:2607.13464](https://arxiv.org/abs/2607.13464)

Release 1.0.2 matches the revised version of the paper, and the table and
section names below refer to that version. Release 1.0.1
([10.5281/zenodo.21361445](https://doi.org/10.5281/zenodo.21361445))
matches the first version, arXiv:2607.13464v1. `CHANGELOG.md` lists the
differences.

The artifact builds the paper's reduced one-round fast-phase games as
semidefinite programs (SDPs), solves the distance-fraud and mafia-fraud
programs for the discrete-variable QDB protocols, and checks the
discrete-variable mafia-fraud values with solver-free exact-arithmetic
primal and dual certificates. For the CV-QDB protocol it evaluates the
paper's closed-form values, including the bright-probe mafia-fraud
attack. The shipped NumPy files retain the raw numerical solver witnesses
for reproducibility.

## Installation

Requires Python 3.10 or newer. With [uv](https://docs.astral.sh/uv/):

```
uv sync
```

This creates a virtual environment from the pinned `uv.lock`. The
environment is large (under 1 GB, mainly PyTorch). Plain
`pip install -e .` works as well but does not use the pinned versions.

Protocol keys used by all commands: `qdb2019` (QDB 2019),
`mutual-entanglement` (Mutual QDB), `e91` (E91 QDB in the paper's fixed
form, in which the prover does not announce its measurement settings),
`cv` (CV-QDB), and `qdb2017` (QDB 2017, treated in the Methods section
"QDB 2017 one-round values").

## Reproducing the paper's values

**Benchmark table (Table 1).**

```
uv run qdb-benchmark table
```

Solves the DF and MF programs for the DV protocols and prints the values,
together with the honest-run completeness of each protocol, as JSON.
The DV values are the Table 1 entries (and the QDB 2017 values of the
Methods). The CV entries are closed forms: the exact DF value, the honest
pass probability, and the success probability of the bright-probe attack.
This attack value bounds the MF value from below, and Table 1 lists the
entry as at least 0.9901. Takes a few seconds on a laptop.

**Exact discrete-variable mafia-fraud certificates (Tables 1 to 3).**

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
E91 QDB, fixed, value-check branches   (8 + sqrt(24 + 17 sqrt(2))) / 16
```

The rational cases use `fractions.Fraction`; QDB 2019 is checked over
`Q(sqrt(5))`; and E91 is checked over
`Q(sqrt(2), sqrt(24 + 17 sqrt(2)))`. Positivity is established by exact
LDL decompositions. The E91 verifier decides algebraic signs using
rational square-root enclosures. In its output, the retained branches
are the value-check branches of the fixed E91 QDB protocol.

**Numerical solver witnesses.**

```
uv run qdb-benchmark certificate --protocol qdb2019 --check-only
```

(similarly for the other DV protocol keys). This NumPy command rebuilds
the tester and reports the residuals, eigenvalues, and duality gap of the
stored solver output. Its floating-point repair and candidate bracket
are diagnostics, not a mathematical certificate; use
`exact-certificate` for the proof.

**CV-QDB values (Table 1 and the section "Continuous-variable QDB").**

```
uv run qdb-benchmark cv-values
```

Evaluates the paper's closed forms for CV-QDB with the Python standard
library only. The model uses `hbar = 2` (vacuum variance 1), squeezing
`r = 1.2`, and tolerance `delta = 1.1`, and the honest reply carries its
measured value exactly. The command gives

```
honest pass probability       erf(delta exp(r) / 2)             0.99019
exact DF value                erf(delta / sqrt(2 cosh(2r)))     0.3592
bright-probe MF attack with probe amplitudes A = 10.4 and 2A = 20.8
  amplitude mix-up            (1 - erf(A / 4)) / 2              1.18e-4
  pass, wrong quadrature      erf(delta / (2 sqrt(cosh(2r))))   0.2586
  attack success probability                                    0.990103
  probe photons               5 A^2 / 4                         135
  challenge-mode photons      sinh(r)^2                         2.28
```

The attack success probability bounds the MF value from below. Table 1
lists the entry as at least 0.9901, which is the honest pass probability
rounded down. The attack approaches the honest pass probability as the
probe gets brighter and reaches 0.9901 with the amplitudes above. The
flags `--squeezing`, `--tolerance`, and `--amplitude` change the
parameters, and `--json` prints all values. The module also runs without
the environment, as `python3 verification/cv_closed_forms.py`.

**Single protocol.**

```
uv run qdb-benchmark sdp --protocol e91
```

Solves one DV protocol's programs (`--solver SCS` or `--solver CLARABEL`).
For `--protocol cv`, see the last item of this section.

**Regenerating the numerical solver witnesses.**

```
uv run qdb-benchmark certificate --protocol qdb2019
```

Re-solves the primal and dual MF comb SDPs, saves fresh numerical
witnesses to `certificates/`, and checks their floating-point residuals.

**CV estimate of the paper's first version.**

```
uv run qdb-benchmark sdp --protocol cv
uv run qdb-benchmark cv-sweep
```

The first version of the paper (arXiv:2607.13464v1) estimated the CV-QDB
MF value by a grid search over Gaussian attacks whose probe has at most
two photons on average, which gave 0.9138, and showed a calibration sweep
in its Figure 2. The revised paper uses neither. The published protocol
does not bound the probe energy, and the bright-probe attack above
succeeds with probability at least 0.9901. The estimator is kept for
readers of the first version. The first command prints the estimate (grid
parameters are exposed as `--cv-*` flags, see `--help`; about a minute).
The second reruns the sweep and writes
`results/cv_calibration_sweep.{csv,json}` and
`figures/cv_calibration_sweep.{png,pdf}` (about three minutes;
`--grid-profile quick` runs a coarser grid). The stored outputs of the
sweep are in release 1.0.1.

## Layout

- `qdb_benchmark_standalone.py` — numerical protocol models, SDP
  construction, benchmark routines, and the CLI.
- `verification/` — standalone exact primal/dual certificate verifiers,
  the combined `exact-certificate` entry point, and the closed-form CV-QDB
  values (`cv_closed_forms.py`).
- `tests/` — exact-tester regression checks against the numerical
  protocol implementation, end-to-end witness checks, and checks of the
  CV-QDB values.
- `certificates/` — shipped numerical MF solver witnesses (`.npz` files
  storing the primal comb, probe state, dual point, and protocol metadata).
- `CHANGELOG.md` — changes between releases.

Run the regression suite with:

```
uv run python -m unittest discover -v
```

## License

MIT, see `LICENSE`.

## Citing

To cite the paper:

```bibtex
@misc{bogner2026sdp,
  title         = {{Security evaluation of quantum distance-bounding
                    protocols via semidefinite programming}},
  author        = {Bogner, Kevin and Abidin, Aysajan and
                   Singel{\'e}e, Dave and Preneel, Bart},
  howpublished  = {arXiv:2607.13464},
  eprint        = {2607.13464},
  archiveprefix = {arXiv},
  primaryclass  = {quant-ph},
  year          = {2026},
  doi           = {10.48550/arXiv.2607.13464}
}
```

To cite this artifact itself, use the version DOI of the release you
used, which its Zenodo record lists, or the concept DOI
[10.5281/zenodo.21307978](https://doi.org/10.5281/zenodo.21307978), which
resolves to the latest release.
