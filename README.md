# qdb-benchmark

Code and data artifact for the paper

> **Security evaluation of quantum distance-bounding protocols via
> semidefinite programming**
> Kevin Bogner, Aysajan Abidin, Dave Singelée, Bart Preneel

The artifact builds the paper's reduced one-round fast-phase games as
semidefinite programs (SDPs), solves the distance-fraud and mafia-fraud
programs for the discrete-variable QDB protocols, runs the restricted
Gaussian finite-grid estimator for the CV-QDB protocol, and checks the
shipped mafia-fraud certificates with a solver-independent NumPy
checker.

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

**Certified mafia-fraud brackets (Tables 3 and 4).**

```
uv run qdb-benchmark certificate --protocol qdb2019 --check-only
```

(similarly for the other DV protocol keys). Using NumPy linear algebra
only — no SDP solver — the checker rebuilds the averaged tester from the
protocol description, verifies the stored primal and dual witnesses,
turns them into exactly feasible ones (the exact-feasibility step of
the paper's Appendix A), and prints the certified bracket for the MF
value. This takes seconds per protocol.

Note that the objectives stored in the certificate files are raw solver
output, feasible only up to solver tolerance. The values reported in
the paper are the certified brackets printed by the checker.

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

**Regenerating the certificates.**

```
uv run qdb-benchmark certificate --protocol qdb2019
```

Re-solves the primal and dual MF comb SDPs, saves a fresh certificate
to `certificates/`, and checks it.

## Layout

- `qdb_benchmark_standalone.py` — all code in one self-contained file
  by design; internal sections cover the quantum helpers, the protocol
  models, the SDP construction and certificates, and the CLI.
- `certificates/` — the shipped MF certificates (`.npz` files storing
  the primal comb, probe state, dual witness, and protocol metadata).
- `results/`, `figures/` — output of the CV calibration sweep as used
  in the paper.

## License

MIT, see `LICENSE`.

## Citing

A citation entry will be added once the paper is public.
