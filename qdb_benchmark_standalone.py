#!/usr/bin/env python3
"""Standalone QDB benchmark artifact.

This file mechanically combines the artifact package modules and executable
scripts into one self-contained Python entry point. It intentionally preserves
the original numerical routines and command-line defaults.

Requires Python >= 3.10, NumPy >= 2.0 (``np.trapezoid``), CVXPY >= 1.3
(``cp.kron`` with a variable first argument), PennyLane, and PyTorch. Exact
versions used for the reported results are pinned in ``uv.lock``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pennylane as qml
import torch

try:
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no cover - optional CV sweep plotting dependencies
    pd = None
    sns = None
    plt = None


def _require_cv_sweep_dependencies() -> None:
    missing = []
    if pd is None:
        missing.append("pandas")
    if sns is None:
        missing.append("seaborn")
    if plt is None:
        missing.append("matplotlib")
    if missing:
        raise RuntimeError(
            "The cv-sweep subcommand requires optional plotting dependencies: "
            + ", ".join(missing)
        )


__version__ = "1.0.2"

# === quantum.py ===

"""Shared quantum primitives for the QDB benchmark artifact.

This module contains only low-level state, basis, and Bell-pair utilities.
It does not construct protocol instances or solve optimization problems.
"""


# Global Configuration & Constants
# --------------------------------------------------------------------------- #

TORCH_COMPLEX = torch.complex128
TORCH_FLOAT = torch.float64
torch.set_default_dtype(TORCH_FLOAT)

# --- Shared Basis States ---

# Z-Basis (Computational, Index 0)
Z0_T = torch.tensor([1.0, 0.0], dtype=TORCH_COMPLEX)
Z1_T = torch.tensor([0.0, 1.0], dtype=TORCH_COMPLEX)

# X-Basis (Hadamard, Index 1)
X0_T = torch.tensor([1.0, 1.0], dtype=TORCH_COMPLEX) / np.sqrt(2.0)
X1_T = torch.tensor([1.0, -1.0], dtype=TORCH_COMPLEX) / np.sqrt(2.0)

# W-Basis (Index 2): theta = pi/4 (CHSH A)
_w_angle = np.pi / 4
W0_T = torch.tensor([np.cos(_w_angle / 2), np.sin(_w_angle / 2)], dtype=TORCH_COMPLEX)
W1_T = torch.tensor([-np.sin(_w_angle / 2), np.cos(_w_angle / 2)], dtype=TORCH_COMPLEX)

# V-Basis (Index 3): theta = -pi/4 (CHSH B)
_v_angle = -np.pi / 4
V0_T = torch.tensor([np.cos(_v_angle / 2), np.sin(_v_angle / 2)], dtype=TORCH_COMPLEX)
V1_T = torch.tensor([-np.sin(_v_angle / 2), np.cos(_v_angle / 2)], dtype=TORCH_COMPLEX)

# Global Tensor Registry
TENSOR_BASES = {0: [Z0_T, Z1_T], 1: [X0_T, X1_T], 2: [W0_T, W1_T], 3: [V0_T, V1_T]}

_QUBIT_DEVICE_NAME: str | None = None


def _resolve_qubit_device_name() -> str:
    """
    Prefer the compiled lightning backend for throughput and fall back gracefully.
    """
    global _QUBIT_DEVICE_NAME
    if _QUBIT_DEVICE_NAME is not None:
        return _QUBIT_DEVICE_NAME

    for name in ("lightning.qubit", "default.qubit"):
        try:
            qml.device(name, wires=1)
            _QUBIT_DEVICE_NAME = name
            if name != "lightning.qubit":
                print(
                    "[warn] lightning.qubit unavailable; using default.qubit (slower)."
                )
            return name
        except Exception:
            continue

    raise RuntimeError(
        "No PennyLane qubit simulator available (tried lightning.qubit, default.qubit)."
    )


def _make_qubit_device(wires: int):
    return qml.device(_resolve_qubit_device_name(), wires=wires)


# --------------------------------------------------------------------------- #
# Quantum Helpers
# --------------------------------------------------------------------------- #


def measure_prob_torch(state: torch.Tensor, basis_vector: torch.Tensor) -> torch.Tensor:
    amp = torch.vdot(basis_vector, state)
    return torch.abs(amp) ** 2


def branch_prob_after_measurement(
    ent_state_2q: torch.Tensor,
    measured_basis_idx: int,
    outcome: int,
) -> float:
    """
    Probability of obtaining `outcome` when measuring wire 0 of a 2-qubit pure
    state in basis `measured_basis_idx`.
    """
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")

    U = _basis_unitary_from_tensor_bases(measured_basis_idx)
    Udag = torch.conj(U).T.contiguous()
    identity = torch.eye(2, dtype=TORCH_COMPLEX)
    rotated = torch.kron(Udag, identity) @ ent_state_2q.contiguous()
    psi = rotated.reshape(2, 2)
    v = psi[outcome, :]
    return float(torch.linalg.norm(v).item() ** 2)


# --------------------------------------------------------------------------- #
# Continuous-Variable (CV) Gaussian helpers for CV-QDB
# --------------------------------------------------------------------------- #


def _gaussian_prob_within_tol(var: torch.Tensor, tol: torch.Tensor) -> torch.Tensor:
    """
    If Δ ~ Normal(0, var), then P(|Δ| <= tol) = erf(tol / sqrt(2 var)).
    """
    eps = torch.tensor(1e-12, dtype=TORCH_FLOAT)
    return torch.erf(tol / torch.sqrt(2.0 * (var + eps)))


def _try_get_tmsv_cov_from_pennylane(
    r: float, hbar: float, phase: float = 0.0
) -> torch.Tensor | None:
    """
    Use PennyLane's default.gaussian backend to prepare a two-mode squeezed vacuum (TMSV)
    and read out the 4x4 covariance matrix.

    Ordering used by default.gaussian's reduced_state is:
      (x_0, x_1, p_0, p_1)

    Returns a torch.float64 tensor cov[4,4] or None if the backend is unavailable.
    """
    try:
        dev = qml.device("default.gaussian", wires=2, shots=None, hbar=hbar)

        @qml.qnode(dev)
        def _prep():
            qml.TwoModeSqueezing(r, phase, wires=[0, 1])
            return qml.expval(qml.Identity(wires=0))  # dummy measurement

        _prep()
        cov, _means = dev.reduced_state(wires=[0, 1])
        cov_np = np.array(cov, dtype=np.float64)
        return torch.tensor(cov_np, dtype=TORCH_FLOAT)
    except Exception as e:
        print(
            "[warn] default.gaussian unavailable or failed; falling back to analytic TMSV cov. "
            f"({e})"
        )
        return None


def _tmsv_cov_fallback(r: float, hbar: float) -> torch.Tensor:
    """
    Analytic fallback covariance matrix of the standard TMSV (phase=0), in ordering (x0,x1,p0,p1).
    With v0=hbar/2:
      Var(x0)=Var(x1)=Var(p0)=Var(p1)=v0 cosh(2r)
      Cov(x0,x1)=+v0 sinh(2r)
      Cov(p0,p1)=-v0 sinh(2r)
    """
    v0 = hbar / 2.0
    ch = np.cosh(2.0 * r)
    sh = np.sinh(2.0 * r)

    cov = np.zeros((4, 4), dtype=np.float64)
    # variances
    cov[0, 0] = v0 * ch  # x0
    cov[1, 1] = v0 * ch  # x1
    cov[2, 2] = v0 * ch  # p0
    cov[3, 3] = v0 * ch  # p1

    # correlations
    cov[0, 1] = cov[1, 0] = v0 * sh  # x0-x1 correlated
    cov[2, 3] = cov[3, 2] = -v0 * sh  # p0-p1 anti-correlated (standard convention)

    return torch.tensor(cov, dtype=TORCH_FLOAT)


# --------------------------------------------------------------------------- #
# Entanglement Helpers (2-qubit) for E91-style protocols
# --------------------------------------------------------------------------- #

_ENTANGLED_STATE_CACHE: Dict[str, torch.Tensor] = {}


def _basis_unitary_from_tensor_bases(basis_idx: int) -> torch.Tensor:
    """
    Build a 2x2 unitary U whose columns are the basis vectors in TENSOR_BASES[basis_idx].
    That is, U|0> = |b0>, U|1> = |b1>.
    """
    b0, b1 = TENSOR_BASES[basis_idx]
    U = torch.stack([b0, b1], dim=1).to(dtype=TORCH_COMPLEX)
    return U


def get_bell_state_2q(label: str = "beta00") -> torch.Tensor:
    """
    Prepare a 2-qubit Bell state using PennyLane and return a torch statevector
    in computational ordering: |00>, |01>, |10>, |11>.

    Supported labels (aliases):
      - beta00 / β00 / phi+   : (|00>+|11>)/sqrt(2)  = |Φ+>
      - beta01 / β01 / phi-   : (|00>-|11>)/sqrt(2)  = |Φ->
      - beta10 / β10 / psi+   : (|01>+|10>)/sqrt(2)  = |Ψ+>
      - beta11 / β11 / psi- / singlet : (|01>-|10>)/sqrt(2) = |Ψ->
    """
    raw = label.strip().lower()
    key = (
        raw.replace("ψ", "psi")
        .replace("φ", "phi")
        .replace("β", "beta")
        .replace("−", "-")
        .replace(" ", "")
        .replace("_", "")
    )

    aliases = {
        "beta00": "beta00",
        "b00": "beta00",
        "phi+": "beta00",
        "phiplus": "beta00",
        "phi-plus": "beta00",
        "beta01": "beta01",
        "b01": "beta01",
        "phi-": "beta01",
        "phiminus": "beta01",
        "phi-minus": "beta01",
        "beta10": "beta10",
        "b10": "beta10",
        "psi+": "beta10",
        "psiplus": "beta10",
        "psi-plus": "beta10",
        "beta11": "beta11",
        "b11": "beta11",
        "psi-": "beta11",
        "psiminus": "beta11",
        "psi-minus": "beta11",
        "singlet": "beta11",
    }
    key = aliases.get(key, key)

    cached = _ENTANGLED_STATE_CACHE.get(key)
    if cached is not None:
        return cached

    dev = _make_qubit_device(wires=2)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def _prep_state() -> torch.Tensor:
        # Start in |00>, build |Φ+> then apply local Paulis on wire 1.
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])

        if key == "beta00":
            pass
        elif key == "beta01":
            # (I ⊗ Z)|Φ+> = |Φ->
            qml.PauliZ(wires=1)
        elif key == "beta10":
            # (I ⊗ X)|Φ+> = |Ψ+>
            qml.PauliX(wires=1)
        elif key == "beta11":
            # (I ⊗ XZ)|Φ+> = |Ψ-> (global phase irrelevant)
            qml.PauliZ(wires=1)
            qml.PauliX(wires=1)
        else:
            raise ValueError(f"Unknown Bell state label: {label!r}")

        return qml.state()

    with torch.no_grad():
        st = _prep_state()
        if not isinstance(st, torch.Tensor):
            st = torch.tensor(st, dtype=TORCH_COMPLEX)
        st = st.to(dtype=TORCH_COMPLEX).detach()

    _ENTANGLED_STATE_CACHE[key] = st
    return st


def conditional_state_after_measurement(
    ent_state_2q: torch.Tensor,
    measured_basis_idx: int,
    outcome: int,
) -> torch.Tensor:
    """
    Given a 2-qubit pure state |ψ> on wires (0=Verifier, 1=Prover),
    compute the (pure) conditional state on wire 1 after measuring wire 0
    in basis 'measured_basis_idx' and obtaining 'outcome' in {0,1}.

    Implements:
      |ψ_cond> ∝ (⟨b_outcome| ⊗ I) |ψ>
    where {|b0>,|b1>} is the measurement basis.

    Returned state is a 2-vector in the *computational representation* of wire 1.
    """
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")

    # Rotate measurement basis -> Z on wire 0 by applying U† on wire 0.
    U = _basis_unitary_from_tensor_bases(measured_basis_idx)
    Udag = torch.conj(U).T.contiguous()
    identity = torch.eye(2, dtype=TORCH_COMPLEX)

    rotated = torch.kron(Udag, identity) @ ent_state_2q.contiguous()

    # Reshape: rows correspond to wire0 outcome, columns correspond to wire1 amplitudes
    psi = rotated.reshape(2, 2)
    v = psi[outcome, :].clone()

    norm = torch.linalg.norm(v)
    if norm.item() < 1e-12:
        raise RuntimeError("Post-measurement state has ~0 norm (unexpected).")

    return (v / norm).to(dtype=TORCH_COMPLEX)


class BellPairCollapseModel:
    """
    Utility that:
      - prepares a fixed Bell state |β> (via PennyLane)
      - precomputes the prover-side collapsed 1-qubit state after measuring
        the verifier-side qubit in any basis in `basis_ids`, for outcomes {0,1}
      - also infers the deterministic prover measurement bit in that same basis.

    This matches the entanglement premise: after measuring its half, the verifier can
    predict what the prover will obtain if it measures in the same setting.
    """

    def __init__(self, bell_state: str = "beta00", basis_ids: List[int] | None = None):
        self.bell_state = bell_state
        self.ent_state_2q = get_bell_state_2q(bell_state)

        if basis_ids is None:
            basis_ids = list(TENSOR_BASES.keys())

        self.collapsed: Dict[Tuple[int, int], torch.Tensor] = {}
        self.prover_bit: Dict[Tuple[int, int], int] = {}

        with torch.no_grad():
            for a_idx in basis_ids:
                for c in (0, 1):
                    psi_p = conditional_state_after_measurement(
                        self.ent_state_2q,
                        measured_basis_idx=a_idx,
                        outcome=c,
                    ).detach()

                    self.collapsed[(a_idx, c)] = psi_p

                    p0 = measure_prob_torch(psi_p, TENSOR_BASES[a_idx][0]).item()
                    p1 = measure_prob_torch(psi_p, TENSOR_BASES[a_idx][1]).item()
                    bit = 0 if p0 >= p1 else 1

                    if max(p0, p1) < 1.0 - 1e-8:
                        print(
                            f"[warn] Bell={bell_state}, basis={a_idx}, outcome={c}: "
                            f"prover measurement not deterministic (p0={p0:.6f}, p1={p1:.6f})"
                        )

                    self.prover_bit[(a_idx, c)] = bit

    def challenge_state_for_prover(
        self, a_idx: int, verifier_outcome: int
    ) -> torch.Tensor:
        return self.collapsed[(a_idx, verifier_outcome)]

    def expected_prover_measurement_bit(self, a_idx: int, verifier_outcome: int) -> int:
        return self.prover_bit[(a_idx, verifier_outcome)]


# --- NumPy density-matrix helpers ---


def state_to_np(vec) -> np.ndarray:
    """Convert a Torch/PennyLane state vector to a complex NumPy array."""
    if hasattr(vec, "detach"):
        return np.asarray(vec.detach().cpu().numpy(), dtype=np.complex128)
    return np.asarray(vec, dtype=np.complex128)


def projector(ket: np.ndarray) -> np.ndarray:
    """Return |ket><ket| as a complex NumPy matrix."""
    return np.outer(ket, np.conj(ket))


# Backward-compatible private aliases used by the original manuscript scripts.
_state_to_np = state_to_np
_projector = projector

# === protocols.py ===

"""Protocol models for the reduced one-round QDB benchmark games.

The classes in this module describe branch variables, challenge states,
acceptance targets, and honest pre-ask response channels.  They intentionally
model the reduced games reported in the paper, not full protocol security.
"""


class QDBProtocol(ABC):
    """Abstract base class for QDB protocols."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def iterate_rounds(self) -> Iterator[Dict[str, int]]:
        """
        Yields dictionaries representing the ground truth of every valid session.
        E.g., {'a': 0, 'b': 0}
        """
        pass

    def iterate_attack_rounds(self) -> Iterator[Dict[str, int]]:
        """
        Yields the per-round configurations used by the reduced SDP attack games.
        Protocols with richer session-level checks can override this to keep the
        attack abstraction narrower than the honest session model.
        """
        yield from self.iterate_rounds()

    def get_challenge_state(self, config: Dict[str, int], c: int) -> torch.Tensor:
        """Returns the quantum state sent by Verifier based on basis 'a'."""
        basis_idx = config["a"]
        return TENSOR_BASES[basis_idx][c]

    def get_response_state(self, config: Dict[str, int], c: int) -> torch.Tensor:
        """
        Returns the verifier's target response state for acceptance testing.
        Default assumes 'b' is the response basis.
        """
        basis_idx = config["b"]
        return TENSOR_BASES[basis_idx][c]

    def get_honest_response_states(
        self, config: Dict[str, int], c: int
    ) -> List[Tuple[float, torch.Tensor]]:
        """
        Distribution over the honest prover's response states in an actual round.

        Default DV behavior:
          - measure the received challenge in basis 'a'
          - re-encode the observed bit in basis 'b' if present, otherwise in 'a'

        Protocols with extra round logic, such as XOR masks, should override this.
        """
        a_idx = config["a"]
        response_basis_idx = config.get("b", a_idx)
        challenge_state = self.get_challenge_state(config, c)

        dist: List[Tuple[float, torch.Tensor]] = []
        for bit in (0, 1):
            prob = float(measure_prob_torch(challenge_state, TENSOR_BASES[a_idx][bit]))
            if prob <= 1e-12:
                continue
            dist.append((prob, TENSOR_BASES[response_basis_idx][bit]))

        if not dist:
            raise RuntimeError("Honest response distribution is empty.")
        return dist

    def get_preask_response_states(
        self, config: Dict[str, int], c: int
    ) -> List[Tuple[float, torch.Tensor]]:
        """
        Distribution over what the honest Prover would send if queried with a
        fake challenge bit c (pre-ask). Default: deterministic, tied to config.
        """
        return [(1.0, self.get_response_state(config, c))]

    def get_completeness_override(self) -> Tuple[str, float] | None:
        return None

    def get_honest_session_summary(self) -> Dict[str, object] | None:
        return None


class QDB2017(QDBProtocol):
    """
    QDB 2017 with a reduced rapid phase plus a session-level final MAC check.

    The current SDP attack games still target only the rapid response round.
    The final MAC is modeled in honest-session completeness/reporting, not as a
    multi-message attack game.
    """

    name = "QDB 2017"

    def iterate_rounds(self) -> Iterator[Dict[str, int]]:
        for a in [0, 1]:
            yield {"a": a}

    def get_response_state(self, config: Dict[str, int], c: int) -> torch.Tensor:
        basis_idx = config["a"]
        return TENSOR_BASES[basis_idx][c]

    def get_honest_session_summary(self) -> Dict[str, object]:
        rapid_accept_probs: List[float] = []
        transcript_bits: List[int] = []

        for cfg in self.iterate_rounds():
            for c in (0, 1):
                target = _projector(_state_to_np(self.get_response_state(cfg, c)))
                accept = 0.0
                for weight, honest_state in self.get_honest_response_states(cfg, c):
                    honest_proj = _projector(_state_to_np(honest_state))
                    accept += float(weight) * float(
                        np.real(np.trace(target @ honest_proj))
                    )
                rapid_accept_probs.append(accept)
                transcript_bits.append(int(c))

        rapid_phase_success = (
            float(np.mean(rapid_accept_probs)) if rapid_accept_probs else float("nan")
        )
        rapid_phase_pass = all(abs(v - 1.0) <= 1e-8 for v in rapid_accept_probs)

        # In this reduced model the MAC is an ideal final consistency check over
        # transcript names c_P (prover decoded) and c_V (verifier decoded).
        final_mac_success = 1.0
        final_mac_pass = True
        session_acceptance = 1.0 if rapid_phase_pass and final_mac_pass else 0.0

        return {
            "phase": "verification",
            "stores_decoded_transcript": True,
            "transcript_length_bits": len(transcript_bits),
            "rapid_phase_success": rapid_phase_success,
            "rapid_phase_pass": rapid_phase_pass,
            "final_mac_mode": (
                "tau = MAC_x(ID_verifier, ID_prover, N_v, N_p, c_P); "
                "verifier recomputes with c_V"
            ),
            "final_mac_inputs": [
                "verifier_id",
                "prover_id",
                "verifier_nonce",
                "prover_nonce",
                "prover_decoded_transcript_c_P",
            ],
            "final_mac_verification": (
                "tau == MAC_x(ID_verifier, ID_prover, N_v, N_p, c_V)"
            ),
            "rapid_phase_accept_condition": ("all c_i == c_V_i and Delta t_i <= t_max"),
            "transcript_names": {
                "prover_decoded": "c_P",
                "verifier_decoded": "c_V",
            },
            "final_mac_success": final_mac_success,
            "final_mac_pass": final_mac_pass,
            "session_acceptance": session_acceptance,
            "attack_round_templates": sum(1 for _ in self.iterate_attack_rounds()),
            "full_round_templates": sum(1 for _ in self.iterate_rounds()),
            "attack_scope": "reduced rapid-phase subset only; final MAC not in SDP game",
        }

    def get_completeness_override(self) -> Tuple[str, float] | None:
        summary = self.get_honest_session_summary()
        return "exact-session", float(summary["session_acceptance"])


class QDB2019(QDBProtocol):
    """QDB 2019: The response basis 'b' is chosen independent of 'a'."""

    name = "QDB 2019 (independent)"

    def iterate_rounds(self) -> Iterator[Dict[str, int]]:
        for a in [0, 1]:
            for b in [0, 1]:
                yield {"a": a, "b": b}


class CVQDB(QDBProtocol):
    """
    Continuous-Variable (Gaussian) QDB protocol (Infocom'25).

    Core protocol (per round i):
      - Verifier prepares EPR (modes A,B), sends B.
      - Verifier measures mode A in x if a_i=0 else p, records m_i.
      - Prover measures mode B in same quadrature (a_i), records m'_i.
      - Prover encodes g(a_i) * m'_i into mode C's x if b_i=0 else p, where
        g(0)=+1 and g(1)=-1 to compensate the TMSV p-quadrature anti-correlation.
      - Verifier measures mode C in quadrature selected by b_i and checks m_i ≈ m''_i.

    IMPORTANT:
      This runner’s default interface is single-qubit states. CV-QDB is continuous-valued, so we
      mark this protocol as quantum_model="cv" and route it to CV-specific simulators.
    """

    quantum_model = "cv"
    name = "CV-QDB (Gaussian)"

    def __init__(
        self,
        epr_r: float = 1.2,  # squeezing strength (higher -> stronger correlations)
        epr_phase: float = 0.0,  # TwoModeSqueezing phase
        hbar: float = 2.0,  # default.gaussian convention: vacuum Var(x)=Var(p)=hbar/2
        accept_tol: float = 1.1,  # "acceptable tolerance" for |m - m''|
        response_var: float
        | None = None,  # verifier homodyne noise variance on response mode C
        use_pennylane_stats: bool = True,
    ):
        self.epr_r = float(epr_r)
        self.epr_phase = float(epr_phase)
        self.hbar = float(hbar)

        self.accept_tol = torch.tensor(float(accept_tol), dtype=TORCH_FLOAT)

        # Default to noiseless response measurement to match the DV protocol baselines.
        if response_var is None:
            response_var = 0.0
        self.response_var = torch.tensor(float(response_var), dtype=TORCH_FLOAT)

        cov = None
        if use_pennylane_stats:
            cov = _try_get_tmsv_cov_from_pennylane(
                self.epr_r, self.hbar, self.epr_phase
            )
        if cov is None:
            cov = _tmsv_cov_fallback(self.epr_r, self.hbar)

        # Store covariance in ordering (x_A, x_B, p_A, p_B)
        self.cov = cov.detach()

    def iterate_rounds(self) -> Iterator[Dict[str, int]]:
        for a in [0, 1]:
            for b in [0, 1]:
                yield {"a": a, "b": b}

    # DV interface methods are not used for CV protocols:
    def get_challenge_state(self, config: Dict[str, int], c: int) -> torch.Tensor:
        raise NotImplementedError(
            "CVQDB is continuous-variable; use CV simulators (quantum_model='cv')."
        )

    def get_response_state(self, config: Dict[str, int], c: int) -> torch.Tensor:
        raise NotImplementedError(
            "CVQDB is continuous-variable; use CV simulators (quantum_model='cv')."
        )

    # ---- CV moment helpers ----

    def _m_index(self, a: int) -> int:
        # cov ordering: (x_A, x_B, p_A, p_B)
        return 0 if a == 0 else 2

    def _m_prime_index(self, a: int) -> int:
        return 1 if a == 0 else 3

    def var_m(self, a: int) -> torch.Tensor:
        idx = self._m_index(a)
        return self.cov[idx, idx]

    def var_m_prime(self, a: int) -> torch.Tensor:
        idx = self._m_prime_index(a)
        return self.cov[idx, idx]

    def cov_m_m_prime(self, a: int) -> torch.Tensor:
        return self.cov[self._m_index(a), self._m_prime_index(a)]

    def honest_response_gain(self, a: int, b: int) -> torch.Tensor:
        """
        Gain applied when the prover writes its response into the quadrature selected by b.

        In the current Gaussian abstraction, b selects whether the response is written
        into x_C or p_C, while the sign correction depends on the verifier/prover
        measurement quadrature a.
        """
        _ = b
        return torch.tensor(1.0 if a == 0 else -1.0, dtype=TORCH_FLOAT)

    def response_measurement_var(self, b: int) -> torch.Tensor:
        """
        Verifier measurement noise variance on the response quadrature selected by b.
        """
        _ = b
        return self.response_var

    def _y_coeff(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Attacker measures mode B along quadrature:
          y = cos(theta) * x_B + sin(theta) * p_B
        In ordering (x_A, x_B, p_A, p_B), that is coeff=[0, cosθ, 0, sinθ].
        """
        return torch.stack(
            [
                torch.tensor(0.0, dtype=TORCH_FLOAT),
                torch.cos(theta),
                torch.tensor(0.0, dtype=TORCH_FLOAT),
                torch.sin(theta),
            ],
            dim=0,
        )

    def var_y(self, theta: torch.Tensor) -> torch.Tensor:
        c = self._y_coeff(theta)
        return c @ self.cov @ c

    def cov_m_y(self, a: int, theta: torch.Tensor) -> torch.Tensor:
        m_idx = self._m_index(a)
        c = self._y_coeff(theta)
        return self.cov[m_idx, :] @ c


class QDBMutualEntanglementFull(QDBProtocol):
    """
    Mutual QDB with both timed legs and final opening.

    Session structure modeled here:
      - A sends a Bell-pair half to B; B measures in basis a_i.
      - B sends |m'_i xor r_i> in basis b_i to A.
      - A decodes and combines with its Bell-pair outcome to recover r_i.
      - A sends |r_i> in basis c_i back to B.
      - In the final phase, B opens the commitment to r and reveals m'.

    The current SDP interface is still single-response and therefore only applies
    to the reduced B->A timed leg. Honest-session completeness includes both timed
    legs plus the final opening/check phase.
    """

    name = "Mutual QDB (Full Session)"

    def __init__(self, bell_state: str = "beta00"):
        self.bell_state = bell_state
        self._bell = BellPairCollapseModel(bell_state=bell_state, basis_ids=[0, 1])

    def iterate_rounds(self) -> Iterator[Dict[str, int]]:
        for a in [0, 1]:
            for b in [0, 1]:
                for c_reg in [0, 1]:
                    for r in [0, 1]:
                        yield {"a": a, "b": b, "c_reg": c_reg, "r": r}

    def iterate_attack_rounds(self) -> Iterator[Dict[str, int]]:
        # Reduced attack model: only the first timed B->A leg.
        for a in [0, 1]:
            for b in [0, 1]:
                for r in [0, 1]:
                    yield {"a": a, "b": b, "r": r}

    def get_challenge_state(self, config: Dict[str, int], c: int) -> torch.Tensor:
        return self._bell.challenge_state_for_prover(config["a"], c)

    def get_response_state(self, config: Dict[str, int], c: int) -> torch.Tensor:
        m_prime = self._bell.expected_prover_measurement_bit(config["a"], c)
        return TENSOR_BASES[config["b"]][m_prime ^ config["r"]]

    def get_preask_response_states(
        self, config: Dict[str, int], c: int
    ) -> List[Tuple[float, torch.Tensor]]:
        return [(1.0, TENSOR_BASES[config["b"]][c ^ config["r"]])]

    def get_honest_response_states(
        self, config: Dict[str, int], c: int
    ) -> List[Tuple[float, torch.Tensor]]:
        a_idx = config["a"]
        b_idx = config["b"]
        nonce = config["r"]
        challenge_state = self.get_challenge_state(config, c)

        dist: List[Tuple[float, torch.Tensor]] = []
        for bit in (0, 1):
            prob = float(measure_prob_torch(challenge_state, TENSOR_BASES[a_idx][bit]))
            if prob <= 1e-12:
                continue
            dist.append((prob, TENSOR_BASES[b_idx][bit ^ nonce]))

        if not dist:
            raise RuntimeError("Honest response distribution is empty.")
        return dist

    def get_honest_session_summary(self) -> Dict[str, object]:
        b_to_a_accept_probs: List[float] = []
        a_to_b_accept_probs: List[float] = []
        recovered_r_matches: List[bool] = []
        revealed_m_prime_matches: List[bool] = []

        for cfg in self.iterate_rounds():
            a_idx = cfg["a"]
            b_idx = cfg["b"]
            c_basis_idx = cfg["c_reg"]
            nonce = cfg["r"]

            for verifier_outcome in (0, 1):
                expected_m_prime = self._bell.expected_prover_measurement_bit(
                    a_idx, verifier_outcome
                )

                # Timed leg 1: B -> A
                b_to_a_target = _projector(
                    _state_to_np(TENSOR_BASES[b_idx][expected_m_prime ^ nonce])
                )
                b_to_a_accept = 0.0
                for weight, honest_state in self.get_honest_response_states(
                    cfg, verifier_outcome
                ):
                    honest_proj = _projector(_state_to_np(honest_state))
                    b_to_a_accept += float(weight) * float(
                        np.real(np.trace(b_to_a_target @ honest_proj))
                    )
                b_to_a_accept_probs.append(b_to_a_accept)

                decoded_masked_bit = expected_m_prime ^ nonce
                recovered_r = decoded_masked_bit ^ expected_m_prime
                recovered_r_matches.append(recovered_r == nonce)

                # Timed leg 2: A -> B
                a_to_b_state = TENSOR_BASES[c_basis_idx][recovered_r]
                a_to_b_target = _projector(_state_to_np(a_to_b_state))
                a_to_b_accept_probs.append(
                    float(np.real(np.trace(a_to_b_target @ a_to_b_target)))
                )

                # Final reveal/opening consistency checks.
                revealed_m_prime_matches.append(
                    expected_m_prime
                    == self._bell.expected_prover_measurement_bit(
                        a_idx, verifier_outcome
                    )
                )

        b_to_a_success = (
            float(np.mean(b_to_a_accept_probs)) if b_to_a_accept_probs else float("nan")
        )
        a_to_b_success = (
            float(np.mean(a_to_b_accept_probs)) if a_to_b_accept_probs else float("nan")
        )
        b_to_a_pass = all(abs(v - 1.0) <= 1e-8 for v in b_to_a_accept_probs)
        a_to_b_pass = all(abs(v - 1.0) <= 1e-8 for v in a_to_b_accept_probs)
        commitment_open_success = 1.0 if all(recovered_r_matches) else 0.0
        commitment_open_pass = all(recovered_r_matches)
        m_prime_reveal_success = 1.0 if all(revealed_m_prime_matches) else 0.0
        m_prime_reveal_pass = all(revealed_m_prime_matches)
        session_acceptance = (
            1.0
            if (
                b_to_a_pass
                and a_to_b_pass
                and commitment_open_pass
                and m_prime_reveal_pass
            )
            else 0.0
        )

        return {
            "phase": "verification",
            "timed_legs": ["B->A", "A->B"],
            "b_to_a_success": b_to_a_success,
            "b_to_a_pass": b_to_a_pass,
            "a_to_b_success": a_to_b_success,
            "a_to_b_pass": a_to_b_pass,
            "commitment_open_mode": "ideal final opening consistency check",
            "commitment_open_success": commitment_open_success,
            "commitment_open_pass": commitment_open_pass,
            "m_prime_reveal_success": m_prime_reveal_success,
            "m_prime_reveal_pass": m_prime_reveal_pass,
            "session_acceptance": session_acceptance,
            "attack_round_templates": sum(1 for _ in self.iterate_attack_rounds()),
            "full_round_templates": sum(1 for _ in self.iterate_rounds()),
            "attack_scope": (
                "reduced B->A timed-leg subset only; A->B leg and final opening "
                "not in SDP game"
            ),
        }

    def get_completeness_override(self) -> Tuple[str, float] | None:
        summary = self.get_honest_session_summary()
        return "exact-session", float(summary["session_acceptance"])


class QDBE91(QDBProtocol):
    """
    E91 QDB Protocol (entanglement-based).

    Paper uses a singlet pair (|01>-|10>)/sqrt(2) and specific measurement settings
    and encoding bases. This implementation keeps the simulator’s 1-qubit interface
    by computing the post-measurement (collapsed) state on the Prover side
    conditioned on the Verifier’s measurement basis/outcome.

    The benchmark values refer to the fixed protocol of the paper that this
    artifact accompanies: the prover's setting ap is derived from the PRF, like
    b, and is never announced. In the published protocol the prover announces ap
    after the fast phase, which allows an attack that the one-round game does
    not cover.
    """

    name = "E91 QDB"
    VERIFIER_SETTING_TO_BASIS = {0: 1, 1: 2, 2: 0}  # X, W, Z
    PROVER_SETTING_TO_BASIS = {0: 2, 1: 0, 2: 3}  # W, Z, V
    VALUE_MATCHING_SETTING_PAIRS = {(1, 0), (2, 1)}
    CHSH_SETTING_PAIRS = {(0, 0), (0, 2), (2, 0), (2, 2)}

    def __init__(self, bell_state: str = "singlet"):
        self.bell_state = bell_state

        # Prepare entangled pair once using PennyLane
        self._ent_state_2q = get_bell_state_2q(bell_state)

        # Cache collapsed prover states: (verifier_basis_idx, verifier_outcome) -> |ψ> on prover wire
        self._collapsed_state_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._verifier_outcome_prob_cache: Dict[Tuple[int, int], float] = {}
        with torch.no_grad():
            for basis_idx in TENSOR_BASES.keys():
                for outcome in (0, 1):
                    self._verifier_outcome_prob_cache[(basis_idx, outcome)] = (
                        branch_prob_after_measurement(
                            self._ent_state_2q,
                            measured_basis_idx=basis_idx,
                            outcome=outcome,
                        )
                    )
                    self._collapsed_state_cache[(basis_idx, outcome)] = (
                        conditional_state_after_measurement(
                            self._ent_state_2q,
                            measured_basis_idx=basis_idx,
                            outcome=outcome,
                        ).detach()
                    )

    def iterate_rounds(self) -> Iterator[Dict[str, int]]:
        """
        Yields the full authentication-phase round templates:
          - Table III value-matching rounds
          - Table IV CHSH rounds
          - the remaining no-check rounds
        """
        for av in (0, 1, 2):
            for ap in (0, 1, 2):
                for b in (0, 1):
                    yield {
                        "av": av,
                        "ap": ap,
                        "av_idx": self.VERIFIER_SETTING_TO_BASIS[av],
                        "a": self.PROVER_SETTING_TO_BASIS[ap],
                        "b": b,
                    }

    def iterate_attack_rounds(self) -> Iterator[Dict[str, int]]:
        for cfg in self.iterate_rounds():
            if self._round_check_kind(cfg) == "value_matching":
                yield cfg

    def _round_check_kind(self, config: Dict[str, int]) -> str:
        pair = (config["av"], config["ap"])
        if pair in self.VALUE_MATCHING_SETTING_PAIRS:
            return "value_matching"
        if pair in self.CHSH_SETTING_PAIRS:
            return "chsh"
        return "no_check"

    def get_challenge_state(self, config: Dict[str, int], c: int) -> torch.Tensor:
        """
        In the protocol, the Verifier distributes an entangled pair and measures its half
        in basis av. Conditioning on the Verifier’s outcome c, the Prover’s half collapses
        to a pure state.
        """
        verifier_basis = config["av_idx"]
        return self._collapsed_state_cache[(verifier_basis, c)]

    def get_response_state(self, config: Dict[str, int], c: int) -> torch.Tensor:
        """
        Target response state for the prover's measurement basis/encoding basis pair.

        The current single-qubit response interface models the prover as measuring the
        collapsed entangled qubit in its measurement basis and re-encoding that bit in
        basis b. Derive the bit from the cached collapsed state rather than from the
        Bell-state label string.
        """
        b_idx = config["b"]
        prover_basis = config["a"]
        collapsed = self._collapsed_state_cache[(config["av_idx"], c)]
        probs = [
            float(measure_prob_torch(collapsed, TENSOR_BASES[prover_basis][bit]))
            for bit in (0, 1)
        ]
        if max(probs) < 1.0 - 1e-8:
            raise RuntimeError(
                "E91 value-matching target is not deterministic in the prover basis. "
                "This reduced interface only supports deterministic value-matching rounds."
            )
        prover_bit = 0 if probs[0] >= probs[1] else 1
        return TENSOR_BASES[b_idx][prover_bit]

    def get_preask_response_states(
        self, config: Dict[str, int], c: int
    ) -> List[Tuple[float, torch.Tensor]]:
        """
        For Mafia Fraud: "c" is the result the Prover measured from the probe.
        Honest Prover encodes this result "c" into basis "b".
        """
        b_idx = config["b"]
        return [(1.0, TENSOR_BASES[b_idx][c])]

    def get_honest_session_summary(self) -> Dict[str, object]:
        round_type_counts = {
            "value_matching": 0,
            "chsh": 0,
            "no_check": 0,
        }
        value_accept_probs: List[float] = []

        for cfg in self.iterate_rounds():
            kind = self._round_check_kind(cfg)
            round_type_counts[kind] += 1
            if kind == "value_matching":
                for c in (0, 1):
                    target = _projector(_state_to_np(self.get_response_state(cfg, c)))
                    accept = 0.0
                    for weight, honest_state in self.get_honest_response_states(cfg, c):
                        honest_proj = _projector(_state_to_np(honest_state))
                        accept += float(weight) * float(
                            np.real(np.trace(target @ honest_proj))
                        )
                    value_accept_probs.append(accept)

        chsh_expectations: Dict[str, float] = {}
        for av, ap in sorted(self.CHSH_SETTING_PAIRS):
            av_idx = self.VERIFIER_SETTING_TO_BASIS[av]
            ap_idx = self.PROVER_SETTING_TO_BASIS[ap]
            expectation = 0.0
            for c in (0, 1):
                p_c = self._verifier_outcome_prob_cache[(av_idx, c)]
                collapsed = self._collapsed_state_cache[(av_idx, c)]
                for prover_bit in (0, 1):
                    p_bit = float(
                        measure_prob_torch(collapsed, TENSOR_BASES[ap_idx][prover_bit])
                    )
                    sign = 1.0 if prover_bit == c else -1.0
                    expectation += p_c * p_bit * sign
            chsh_expectations[f"av{av}_ap{ap}"] = expectation

        chsh_value = (
            chsh_expectations["av0_ap0"]
            - chsh_expectations["av0_ap2"]
            + chsh_expectations["av2_ap0"]
            + chsh_expectations["av2_ap2"]
        )
        chsh_magnitude = abs(chsh_value)

        value_matching_success = (
            float(np.mean(value_accept_probs)) if value_accept_probs else float("nan")
        )
        value_matching_pass = all(abs(v - 1.0) <= 1e-8 for v in value_accept_probs)
        chsh_pass = chsh_magnitude > 2.0 + 1e-8
        session_acceptance = 1.0 if value_matching_pass and chsh_pass else 0.0

        return {
            "phase": "authentication",
            # Fixed protocol: both parties derive the prover's setting from the PRF.
            "requires_prover_setting_disclosure": False,
            "round_type_counts": round_type_counts,
            "value_matching_success": value_matching_success,
            "value_matching_pass": value_matching_pass,
            "chsh_expectations": chsh_expectations,
            "chsh_value": float(chsh_value),
            "chsh_magnitude": float(chsh_magnitude),
            "chsh_pass": chsh_pass,
            "session_acceptance": session_acceptance,
            "attack_round_templates": sum(1 for _ in self.iterate_attack_rounds()),
            "full_round_templates": sum(1 for _ in self.iterate_rounds()),
            "attack_scope": "reduced value-matching subset only",
        }

    def get_completeness_override(self) -> Tuple[str, float] | None:
        summary = self.get_honest_session_summary()
        return "exact-session", float(summary["session_acceptance"])


# --------------------------------------------------------------------------- #
# Protocol Key Helpers
# --------------------------------------------------------------------------- #


def _canonical_protocol_key(protocol: str) -> str:
    return protocol.strip().lower()


# === sdp.py ===

"""SDP and Gaussian benchmark routines for the QDB artifact.

This module contains the numerical tasks reported in the paper:
DF state SDPs, DV MF comb SDPs, and numerical MF witness generation/checking.
It also keeps the restricted finite-grid Gaussian CV estimator of the paper's
first version (arXiv:2607.13464v1). The revised paper does not use that
estimator. Its CV-QDB values are closed forms, evaluated in
``verification/cv_closed_forms.py``.
"""


try:
    from scipy import special as scipy_special
except ImportError:  # pragma: no cover - optional vectorized helper
    scipy_special = None

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - runtime dependency gate
    cp = None


REPO_ROOT = Path(__file__).resolve().parent

QubitDensity = np.ndarray
RoundCfg = Dict[str, int]
RoundEntry = Tuple[RoundCfg, QubitDensity, QubitDensity]  # cfg, challenge, target


@dataclass
class SDPSolveResult:
    status: str
    value: float


def _dv_protocol_completeness(protocol: QDBProtocol) -> float:
    vals: List[float] = []
    for cfg in protocol.iterate_rounds():
        for c in (0, 1):
            target = _projector(_state_to_np(protocol.get_response_state(cfg, c)))
            round_accept = 0.0
            for weight, honest_state in protocol.get_honest_response_states(cfg, c):
                honest_proj = _projector(_state_to_np(honest_state))
                round_accept += float(weight) * float(
                    np.real(np.trace(target @ honest_proj))
                )
            vals.append(round_accept)
    return float(np.mean(vals))


def _cv_protocol_completeness(protocol: CVQDB) -> float:
    vals: List[float] = []
    for cfg in protocol.iterate_rounds():
        a = int(cfg["a"])
        b = int(cfg["b"])
        gain = float(protocol.honest_response_gain(a, b).item())
        response_var = float(protocol.response_measurement_var(b).item())
        mismatch_var = (
            float(protocol.var_m(a).item())
            + (gain * gain) * float(protocol.var_m_prime(a).item())
            - 2.0 * gain * float(protocol.cov_m_m_prime(a).item())
            + response_var
        )
        vals.append(
            _cv_success_prob_from_variance(
                max(mismatch_var, 1e-12),
                protocol.accept_tol,
            )
        )
    return float(np.mean(vals))


def _protocol_completeness(protocol: QDBProtocol) -> Tuple[str, float]:
    override = protocol.get_completeness_override()
    if override is not None:
        return override
    if isinstance(protocol, CVQDB):
        return "exact", _cv_protocol_completeness(protocol)
    return "exact", _dv_protocol_completeness(protocol)


def _protocol_known_vars(protocol: str) -> Tuple[List[str], List[str]]:
    p = _canonical_protocol_key(protocol)
    if p == "qdb2017":
        return ["a"], []
    if p == "qdb2019":
        return ["a", "b"], []
    if p == "mutual-entanglement":
        return ["a", "b", "r"], []
    if p == "e91":
        return ["ap", "b"], []
    if p == "cv":
        return ["a", "b"], []
    raise ValueError(f"Unsupported protocol: {protocol!r}")


@lru_cache(maxsize=None)
def _sdp_protocol_instance(protocol: str) -> QDBProtocol | None:
    p = _canonical_protocol_key(protocol)
    if p == "qdb2017":
        return QDB2017()
    if p == "qdb2019":
        return QDB2019()
    if p == "mutual-entanglement":
        return QDBMutualEntanglementFull()
    if p == "e91":
        return QDBE91()
    return None


def _dv_protocol_instance(protocol: str) -> QDBProtocol:
    p = _canonical_protocol_key(protocol)
    if p == "qdb2017":
        return QDB2017()
    if p == "qdb2019":
        return QDB2019()
    if p == "mutual-entanglement":
        return QDBMutualEntanglementFull()
    if p == "e91":
        return QDBE91()
    raise ValueError(f"Unsupported protocol: {protocol!r}")


def _round_challenge_target(
    protocol: str,
    cfg: Dict[str, int],
    c: int,
) -> Tuple[np.ndarray, np.ndarray]:
    protocol_obj = _sdp_protocol_instance(protocol)
    if protocol_obj is None:
        raise ValueError(f"Unsupported protocol: {protocol!r}")
    challenge = _projector(_state_to_np(protocol_obj.get_challenge_state(cfg, c)))
    target = _projector(_state_to_np(protocol_obj.get_response_state(cfg, c)))
    return challenge, target


def build_round_entries(protocol: str) -> List[RoundEntry]:
    """
    Build round entries (cfg, challenge projector, target projector).

    Uses the protocol model directly so SDP support can cover variants with
    extra per-round variables, such as the mutual-entanglement mask r.

    The per-round bit stored as cfg["c"] is the challenge bit for the
    prepare-and-measure protocols; for the entanglement-based protocols it
    plays the role of the verifier's measurement outcome (the paper's m).
    """
    out: List[RoundEntry] = []
    protocol_obj = _sdp_protocol_instance(protocol)
    if protocol_obj is None:
        raise ValueError(f"Unsupported protocol: {protocol!r}")

    for round_cfg_raw in protocol_obj.iterate_attack_rounds():
        round_cfg = dict(round_cfg_raw)
        for c in (0, 1):
            chal, targ = _round_challenge_target(protocol, round_cfg, c)
            cfg = dict(round_cfg)
            cfg["c"] = c
            out.append((cfg, chal, targ))

    return out


def _view_key(cfg: RoundCfg, known_vars: Iterable[str]) -> Tuple[int, ...]:
    kv = list(known_vars)
    if not kv:
        return tuple()
    return tuple(cfg[k] for k in kv)


def _unique_views(
    rounds: List[RoundEntry], known_vars: Iterable[str]
) -> List[Tuple[int, ...]]:
    seen = {_view_key(cfg, known_vars) for cfg, _, _ in rounds}
    return sorted(seen)


def _partial_trace_out_cvx(J, d: int):
    """
    Partial trace over output subsystem for Choi matrix J in ordering:
      input ⊗ output
    """
    rows = []
    for i in range(d):
        row = []
        for j in range(d):
            expr = 0
            for k in range(d):
                expr += J[i * d + k, j * d + k]
            row.append(expr)
        rows.append(row)
    return cp.bmat(rows)


def _partial_trace_last_subsystem_cvx(X, d_last: int):
    """
    Partial trace over the last subsystem for a matrix in
    lexicographic tensor ordering (..., last).
    """
    d_total = X.shape[0]
    if d_total % d_last != 0:
        raise ValueError("Matrix dimension not divisible by d_last.")
    d_rest = d_total // d_last
    rows = []
    for i in range(d_rest):
        row = []
        for j in range(d_rest):
            expr = 0
            for k in range(d_last):
                expr += X[i * d_last + k, j * d_last + k]
            row.append(expr)
        rows.append(row)
    return cp.bmat(rows)


def _partial_trace_after_first_subsystem_cvx(X, d_first: int, d_rest: int):
    """
    Partial trace over all subsystems after the first one.

    The matrix is ordered as first ⊗ rest.
    """
    rows = []
    for i in range(d_first):
        row = []
        for j in range(d_first):
            expr = 0
            for k in range(d_rest):
                expr += X[i * d_rest + k, j * d_rest + k]
            row.append(expr)
        rows.append(row)
    return cp.bmat(rows)


def _partial_trace_last_subsystem_np(X: np.ndarray, d_last: int) -> np.ndarray:
    d_total = X.shape[0]
    if X.shape[0] != X.shape[1] or d_total % d_last != 0:
        raise ValueError("Matrix shape is incompatible with d_last.")
    d_rest = d_total // d_last
    out = np.zeros((d_rest, d_rest), dtype=np.complex128)
    for i in range(d_rest):
        for j in range(d_rest):
            for k in range(d_last):
                out[i, j] += X[i * d_last + k, j * d_last + k]
    return out


def _partial_trace_after_first_subsystem_np(
    X: np.ndarray,
    d_first: int,
    d_rest: int,
) -> np.ndarray:
    if X.shape != (d_first * d_rest, d_first * d_rest):
        raise ValueError("Matrix shape is incompatible with d_first and d_rest.")
    out = np.zeros((d_first, d_first), dtype=np.complex128)
    for i in range(d_first):
        for j in range(d_first):
            for k in range(d_rest):
                out[i, j] += X[i * d_rest + k, j * d_rest + k]
    return out


def solve_df_state_sdp(
    rounds: List[RoundEntry],
    known_vars: List[str],
    solver: str = "SCS",
) -> SDPSolveResult:
    """Exact DF SDP over one-qubit density matrices partitioned by known_vars views."""
    if cp is None:
        raise RuntimeError(
            "cvxpy is required for SDP solving. Install with: uv add cvxpy (or pip install cvxpy)."
        )

    views = _unique_views(rounds, known_vars)
    rho = {v: cp.Variable((2, 2), hermitian=True) for v in views}

    constraints = []
    for v in views:
        constraints.append(rho[v] >> 0)
        constraints.append(cp.trace(rho[v]) == 1)

    expr = 0
    for cfg, _challenge, target in rounds:
        v = _view_key(cfg, known_vars)
        expr += cp.real(cp.trace(target @ rho[v]))
    expr = expr / len(rounds)

    prob = cp.Problem(cp.Maximize(expr), constraints)
    _solve_problem(prob, solver)
    return SDPSolveResult(status=prob.status, value=float(prob.value))


def _solve_problem(prob, solver: str):
    solver = solver.upper()
    if solver == "SCS":
        prob.solve(solver=solver, eps=1e-7, max_iters=50_000, verbose=False)
        return
    if solver == "CLARABEL":
        prob.solve(
            solver=solver,
            max_iter=1_000,
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
            tol_infeas_abs=1e-10,
            tol_infeas_rel=1e-10,
            tol_ktratio=1e-10,
            verbose=False,
        )
        return
    prob.solve(solver=solver, verbose=False)


def _preask_prover_channel_choi(protocol: str, cfg: Dict[str, int]) -> np.ndarray:
    """
    Choi matrix for the pre-ask prover channel N_cfg: probe -> response.

    The probe arrives in the basis cfg["a"] used by the honest prover's
    measurement (the verifier basis for QDB 2017/2019/Mutual QDB, the prover
    basis a_p for E91) with fake bit r_in. The honest prover response
    distribution is delegated to the protocol model so variants with extra
    parameters, such as mutual-entanglement's mask r, are handled uniformly.

    Returned in input⊗output ordering (probe⊗response).
    """
    protocol_obj = _sdp_protocol_instance(protocol)
    if protocol_obj is None:
        raise ValueError(f"Unsupported protocol: {protocol!r}")

    a = cfg["a"]
    J = np.zeros((4, 4), dtype=np.complex128)
    for r_in in (0, 1):
        p_in = _projector(_state_to_np(TENSOR_BASES[a][r_in]))
        rho_out = np.zeros((2, 2), dtype=np.complex128)
        for weight, state in protocol_obj.get_preask_response_states(cfg, r_in):
            rho_out += float(weight) * _projector(_state_to_np(state))
        J += np.kron(p_in.T, rho_out)
    return J


def _mf_comb_average_tester(protocol: str) -> Tuple[np.ndarray, int]:
    """
    Average tester matrix A for the lifted MF comb objective Tr[A S].

    The returned matrix uses the same register ordering as the comb variable:
    [Q, R, C, O], where Q/R are the pre-ask probe/response and C/O are the
    verifier challenge/final forged output.
    """
    d_q = d_r = d_c = d_o = 2
    d_total = d_q * d_r * d_c * d_o
    tester = np.zeros((d_total, d_total), dtype=np.complex128)

    count = 0
    protocol_obj = _sdp_protocol_instance(protocol)
    if protocol_obj is None:
        raise ValueError(f"Unsupported protocol: {protocol!r}")
    for round_cfg_raw in protocol_obj.iterate_attack_rounds():
        round_cfg = dict(round_cfg_raw)
        j_preask = _preask_prover_channel_choi(protocol, round_cfg)
        for c in (0, 1):
            challenge, target = _round_challenge_target(protocol, round_cfg, c)
            tester += np.kron(j_preask.T, np.kron(challenge.T, target))
            count += 1

    if count == 0:
        raise RuntimeError("MF comb tester has no rounds.")
    return tester / count, count


def _mf_comb_primal_solution(
    protocol: str,
    solver: str = "SCS",
) -> Dict[str, object]:
    if cp is None:
        raise RuntimeError(
            "cvxpy is required for SDP solving. Install with: uv add cvxpy (or pip install cvxpy)."
        )

    d_q = d_r = d_c = d_o = 2
    d_total = d_q * d_r * d_c * d_o
    tester, tester_count = _mf_comb_average_tester(protocol)

    # Comb variable on [Q,R,C,O] and first-slot normalization state on Q.
    S = cp.Variable((d_total, d_total), hermitian=True)
    eta = cp.Variable((d_q, d_q), hermitian=True)

    constraints = [S >> 0, eta >> 0, cp.trace(eta) == 1]
    tr_o = _partial_trace_last_subsystem_cvx(S, d_last=d_o)
    constraints.append(tr_o == cp.kron(eta, np.eye(d_r * d_c)))

    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(tester @ S))), constraints)
    _solve_problem(prob, solver)

    return {
        "status": prob.status,
        "value": float(prob.value),
        "S": S.value,
        "eta": eta.value,
        "tester": tester,
        "tester_count": tester_count,
    }


def solve_mf_full_comb_sdp(
    protocol: str,
    solver: str = "SCS",
) -> SDPSolveResult:
    """
    One-round full MF lifted comb SDP for a supported DV protocol.

    Strategy variable:
      S on spaces [Q, R, C, O] with dims [2,2,2,2], where
        Q = probe sent in pre-ask,
        R = prover response to pre-ask,
        C = intercepted verifier challenge,
        O = final forged output to verifier.

    Deterministic 2-slot comb causality constraints:
      S >= 0
      Tr_O[S] = eta_Q ⊗ I_{R,C}
      eta >= 0, Tr[eta] = 1

    Objective (linear in S):
      average_{cfg,c} Tr[ W_{cfg,c} S ]
      with W_{cfg,c} = J(N_cfg)^T ⊗ challenge_cfg,c^T ⊗ target_cfg,c.
    """
    sol = _mf_comb_primal_solution(protocol=protocol, solver=solver)
    return SDPSolveResult(status=str(sol["status"]), value=float(sol["value"]))


def _mf_comb_dual_solution(
    protocol: str,
    solver: str = "SCS",
) -> Dict[str, object]:
    """
    Solve the explicit MF comb dual used for the numerical witnesses.

    With A the average tester, the constraints are kron(Y, I_O) >= A and
    mu I_Q >= Tr_{RC}(Y), so a feasible pair makes mu an upper bound.
    """
    if cp is None:
        raise RuntimeError(
            "cvxpy is required for SDP solving. Install with: uv add cvxpy (or pip install cvxpy)."
        )

    d_q = d_r = d_c = d_o = 2
    d_rest = d_r * d_c
    tester, tester_count = _mf_comb_average_tester(protocol)

    Y = cp.Variable((d_q * d_rest, d_q * d_rest), hermitian=True)
    mu = cp.Variable()
    slack_strategy = cp.kron(Y, np.eye(d_o)) - tester
    trace_rest_y = _partial_trace_after_first_subsystem_cvx(Y, d_q, d_rest)
    slack_normalization = mu * np.eye(d_q) - trace_rest_y

    prob = cp.Problem(
        cp.Minimize(mu),
        [
            slack_strategy >> 0,
            slack_normalization >> 0,
        ],
    )
    _solve_problem(prob, solver)

    return {
        "status": prob.status,
        "value": float(prob.value),
        "Y": Y.value,
        "mu": float(mu.value),
        "tester": tester,
        "tester_count": tester_count,
    }


def _default_mf_certificate_path(protocol: str) -> Path:
    return REPO_ROOT / "certificates" / f"{protocol}_mf_certificate.npz"


def _resolve_certificate_path(path: str | Path) -> Path:
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    return out_path


def save_mf_certificate(
    protocol: str,
    path: str | Path,
    solver: str = "CLARABEL",
    compressed: bool = True,
) -> Dict[str, object]:
    """
    Solve the primal and explicit dual MF comb SDPs and save a NumPy witness.
    """
    p = _canonical_protocol_key(protocol)
    _dv_protocol_instance(p)
    out_path = _resolve_certificate_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    primal = _mf_comb_primal_solution(protocol=p, solver=solver)
    dual = _mf_comb_dual_solution(protocol=p, solver=solver)

    required = {
        "S": primal["S"],
        "eta": primal["eta"],
        "Y": dual["Y"],
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "Numerical witness solve did not produce solution arrays: "
            + ", ".join(missing)
        )

    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(
        out_path,
        S=np.asarray(primal["S"], dtype=np.complex128),
        eta=np.asarray(primal["eta"], dtype=np.complex128),
        Y=np.asarray(dual["Y"], dtype=np.complex128),
        mu=np.array(float(dual["mu"]), dtype=np.float64),
        primal_value=np.array(float(primal["value"]), dtype=np.float64),
        dual_value=np.array(float(dual["value"]), dtype=np.float64),
        dims=np.array([2, 2, 2, 2], dtype=np.int64),
        protocol=np.array(p),
        fraud_type=np.array("MF"),
        solver=np.array(solver.upper()),
        primal_status=np.array(str(primal["status"])),
        dual_status=np.array(str(dual["status"])),
        tester_count=np.array(int(primal["tester_count"]), dtype=np.int64),
    )

    return {
        "path": str(out_path),
        "protocol": p,
        "solver": solver.upper(),
        "primal_status": str(primal["status"]),
        "dual_status": str(dual["status"]),
        "primal_value": float(primal["value"]),
        "dual_value": float(dual["value"]),
    }


def _hermitian_part(X: np.ndarray) -> np.ndarray:
    return (X + np.conj(X.T)) / 2.0


def _min_eig_hermitian(X: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(_hermitian_part(X))))


def repair_mf_witnesses(
    tester: np.ndarray,
    S: np.ndarray,
    eta: np.ndarray,
    Y: np.ndarray,
    mu: float,
    dims: Iterable[int] = (2, 2, 2, 2),
) -> Dict[str, float]:
    """
    Apply a floating-point feasibility repair and return a candidate bracket.

    Dual: shift Y by e1*I so kron(Y, I_O) dominates the tester, then raise mu
    by e2 so mu*I_Q numerically dominates Tr_RC of the shifted Y.

    Primal: clip and renormalize eta, correct the affine comb residual of S
    in floating-point arithmetic against the repaired eta, and mix with the
    uniform comb
    eta ⊗ I_RC ⊗ I_O/d_O with the smallest weight (plus a small pad) that
    numerically restores positive semidefiniteness.

    This routine uses float64 eigensolvers and a heuristic noise guard. Its
    output is a diagnostic, not a mathematical certificate. The exact
    certificate verifiers under 'verification/' provide the proof.
    """
    d_q, d_r, d_c, d_o = [int(v) for v in dims]
    d_rest = d_r * d_c
    tester_h = _hermitian_part(tester)

    # Dual repair.
    y_h = _hermitian_part(Y)
    e1 = max(0.0, -_min_eig_hermitian(np.kron(y_h, np.eye(d_o)) - tester_h))
    y_rep = y_h + e1 * np.eye(d_q * d_rest)
    trace_rest_y_rep = _partial_trace_after_first_subsystem_np(y_rep, d_q, d_rest)
    e2 = max(0.0, -_min_eig_hermitian(mu * np.eye(d_q) - trace_rest_y_rep))
    mu_rep = mu + e2
    repaired_dual_slack_1 = _min_eig_hermitian(np.kron(y_rep, np.eye(d_o)) - tester_h)
    repaired_dual_slack_2 = _min_eig_hermitian(mu_rep * np.eye(d_q) - trace_rest_y_rep)

    # Primal repair.
    eta_h = _hermitian_part(eta)
    eta_clip = eta_h + max(0.0, -_min_eig_hermitian(eta_h)) * np.eye(d_q)
    eta_rep = eta_clip / float(np.real(np.trace(eta_clip)))
    s_h = _hermitian_part(S)
    correction = (
        np.kron(eta_rep, np.eye(d_rest)) - _partial_trace_last_subsystem_np(s_h, d_o)
    ) / d_o
    s_corrected = s_h + np.kron(correction, np.eye(d_o))
    delta = max(0.0, -_min_eig_hermitian(s_corrected))
    s_uniform = np.kron(eta_rep, np.eye(d_rest * d_o)) / d_o
    lam_uniform = _min_eig_hermitian(s_uniform)
    if delta > 0.0:
        pad = 1e-13
        mix_weight = (delta + pad) / (delta + pad + lam_uniform)
    else:
        mix_weight = 0.0
    s_rep = (1.0 - mix_weight) * s_corrected + mix_weight * s_uniform
    lower = float(np.real(np.trace(tester_h @ s_rep)))
    repaired_min_eig_s = _min_eig_hermitian(s_rep)
    repaired_comb_residual = float(
        np.linalg.norm(
            _partial_trace_last_subsystem_np(s_rep, d_o)
            - np.kron(eta_rep, np.eye(d_rest)),
            ord="fro",
        )
    )

    # Outward rounding with a guard for eigensolver noise.
    guard = 1e-12
    scale = 1e10
    candidate_lower = math.floor((lower - guard) * scale) / scale
    candidate_upper = min(1.0, math.ceil((mu_rep + guard) * scale) / scale)

    numerical_repair_passed = bool(
        repaired_dual_slack_1 >= -guard
        and repaired_dual_slack_2 >= -guard
        and repaired_min_eig_s >= -guard
        and repaired_comb_residual <= guard
        and candidate_lower <= candidate_upper
    )

    return {
        "repair_e1": float(e1),
        "repair_e2": float(e2),
        "repair_mix_weight": float(mix_weight),
        "repaired_lower": lower,
        "repaired_upper": float(mu_rep),
        "repaired_min_eig_S": repaired_min_eig_s,
        "repaired_comb_residual": repaired_comb_residual,
        "repaired_dual_slack_1_min_eig": repaired_dual_slack_1,
        "repaired_dual_slack_2_min_eig": repaired_dual_slack_2,
        "candidate_lower": candidate_lower,
        "candidate_upper": candidate_upper,
        "numerical_repair_passed": numerical_repair_passed,
    }


def check_mf_certificate(
    protocol: str,
    path: str | Path,
    tol: float = 1e-8,
    verbose: bool = True,
) -> Dict[str, float | str | bool]:
    """
    Check numerical residuals of a saved MF solver witness using NumPy.

    This is a tolerance-based diagnostic. It does not certify feasibility over
    the reals; use the exact certificate verifiers under 'verification/' for
    the mathematical proof.
    """
    p = _canonical_protocol_key(protocol)
    _dv_protocol_instance(p)
    cert_path = _resolve_certificate_path(path)
    tester, tester_count = _mf_comb_average_tester(p)

    with np.load(cert_path) as cert:
        S = np.asarray(cert["S"], dtype=np.complex128)
        eta = np.asarray(cert["eta"], dtype=np.complex128)
        Y = np.asarray(cert["Y"], dtype=np.complex128)
        mu = float(np.asarray(cert["mu"]))
        stored_primal = float(np.asarray(cert["primal_value"]))
        stored_dual = float(np.asarray(cert["dual_value"]))
        dims = np.asarray(cert["dims"], dtype=np.int64)
        stored_tester_count = (
            int(np.asarray(cert["tester_count"]))
            if "tester_count" in cert.files
            else tester_count
        )
        stored_protocol = (
            str(np.asarray(cert["protocol"]).item()) if "protocol" in cert.files else p
        )
        stored_fraud_type = (
            str(np.asarray(cert["fraud_type"]).item())
            if "fraud_type" in cert.files
            else "MF"
        )

    expected_dims = np.array([2, 2, 2, 2], dtype=np.int64)
    dims_match = bool(np.array_equal(dims, expected_dims))
    d_q, d_r, d_c, d_o = [int(v) for v in expected_dims]
    d_rest = d_r * d_c

    trace_o_s = _partial_trace_last_subsystem_np(S, d_last=d_o)
    comb_residual = float(
        np.linalg.norm(trace_o_s - np.kron(eta, np.eye(d_rest)), ord="fro")
    )
    trace_eta_residual = float(abs(np.trace(eta) - 1.0))

    dual_slack_strategy = np.kron(Y, np.eye(d_o)) - tester
    dual_slack_normalization = mu * np.eye(
        d_q
    ) - _partial_trace_after_first_subsystem_np(Y, d_q, d_rest)

    primal_trace = np.trace(tester @ S)
    primal_value = float(np.real(primal_trace))
    primal_imag_abs = float(abs(np.imag(primal_trace)))
    dual_value = float(mu)
    duality_gap = float(dual_value - primal_value)

    min_eig_s = _min_eig_hermitian(S)
    min_eig_eta = _min_eig_hermitian(eta)
    min_eig_dual_slack_1 = _min_eig_hermitian(dual_slack_strategy)
    min_eig_dual_slack_2 = _min_eig_hermitian(dual_slack_normalization)
    min_dual_slack_eig = min(min_eig_dual_slack_1, min_eig_dual_slack_2)

    hermitian_residual_s = float(np.linalg.norm(S - np.conj(S.T), ord="fro"))
    hermitian_residual_eta = float(np.linalg.norm(eta - np.conj(eta.T), ord="fro"))
    hermitian_residual_y = float(np.linalg.norm(Y - np.conj(Y.T), ord="fro"))

    stored_primal_residual = float(abs(stored_primal - primal_value))
    stored_dual_residual = float(abs(stored_dual - dual_value))
    tester_count_match = stored_tester_count == tester_count
    metadata_match = stored_protocol == p and stored_fraud_type == "MF"

    repair = repair_mf_witnesses(
        tester=tester, S=S, eta=eta, Y=Y, mu=mu, dims=expected_dims
    )

    numerically_consistent = bool(
        dims_match
        and tester_count_match
        and metadata_match
        and min_eig_s >= -tol
        and min_eig_eta >= -tol
        and min_eig_dual_slack_1 >= -tol
        and min_eig_dual_slack_2 >= -tol
        and comb_residual <= tol
        and trace_eta_residual <= tol
        and primal_imag_abs <= tol
        and duality_gap >= -tol
        and duality_gap <= tol
        and stored_primal_residual <= tol
        and stored_dual_residual <= tol
        and hermitian_residual_s <= tol
        and hermitian_residual_eta <= tol
        and hermitian_residual_y <= tol
    )

    result: Dict[str, float | str | bool] = {
        "protocol": p,
        "path": str(cert_path),
        "tol": float(tol),
        "primal_value": primal_value,
        "dual_value": dual_value,
        "duality_gap": duality_gap,
        "abs_duality_gap": abs(duality_gap),
        "min_eig_S": min_eig_s,
        "min_eig_eta": min_eig_eta,
        "comb_residual": comb_residual,
        "trace_eta_residual": trace_eta_residual,
        "min_eig_dual_slack_1": min_eig_dual_slack_1,
        "min_eig_dual_slack_2": min_eig_dual_slack_2,
        "min_dual_slack_eig": min_dual_slack_eig,
        "stored_primal_residual": stored_primal_residual,
        "stored_dual_residual": stored_dual_residual,
        "hermitian_residual_S": hermitian_residual_s,
        "hermitian_residual_eta": hermitian_residual_eta,
        "hermitian_residual_Y": hermitian_residual_y,
        "tester_count_match": tester_count_match,
        "metadata_match": metadata_match,
        "numerically_consistent": numerically_consistent,
        "status": (
            f"numerically consistent within tolerance {tol:g}"
            if numerically_consistent
            else f"numerical check failed tolerance {tol:g}"
        ),
        **repair,
    }

    if verbose:
        print(f"{p.upper()} MF numerical witness check")
        print()
        print(f"numerical witness:          {cert_path}")
        print(f"primal attack value:        {primal_value:.10f}")
        print(f"dual upper bound:           {dual_value:.10f}")
        print(f"duality gap:                {duality_gap:.3e}")
        print()
        print(f"min eig S:                  {min_eig_s:.3e}")
        print(f"min eig eta:                {min_eig_eta:.3e}")
        print(f"comb residual:              {comb_residual:.3e}")
        print(f"trace eta residual:         {trace_eta_residual:.3e}")
        print()
        print(f"min eig dual slack 1:       {min_eig_dual_slack_1:.3e}")
        print(f"min eig dual slack 2:       {min_eig_dual_slack_2:.3e}")
        print(f"min eig dual slack:         {min_dual_slack_eig:.3e}")
        print()
        print("floating-point feasibility diagnostic (not a proof)")
        print(f"dual shift e1:              {repair['repair_e1']:.3e}")
        print(f"dual shift e2:              {repair['repair_e2']:.3e}")
        print(f"primal mix weight:          {repair['repair_mix_weight']:.3e}")
        print(f"repaired primal value:      {repair['repaired_lower']:.12f}")
        print(f"repaired dual bound:        {repair['repaired_upper']:.12f}")
        print(f"repaired min eig S:         {repair['repaired_min_eig_S']:.3e}")
        print(f"repaired comb residual:     {repair['repaired_comb_residual']:.3e}")
        print(
            "repaired dual slack eigs:   "
            f"{repair['repaired_dual_slack_1_min_eig']:.3e}, "
            f"{repair['repaired_dual_slack_2_min_eig']:.3e}"
        )
        print(
            "candidate bracket:          "
            f"[{repair['candidate_lower']:.10f}, "
            f"{repair['candidate_upper']:.10f}]"
            " (non-rigorous)"
        )
        print(
            "numerical repair:           "
            f"{'passed' if repair['numerical_repair_passed'] else 'FAILED'}"
        )
        print()
        print(f"status: {result['status']}")

    return result


def _cv_is_optimal_status(status: str) -> bool:
    return status in {"optimal", "optimal_inaccurate"}


def _cv_success_prob_from_variance(var: float, tol: torch.Tensor) -> float:
    v = max(float(var), 1e-12)
    return float(
        _gaussian_prob_within_tol(
            torch.tensor(v, dtype=TORCH_FLOAT),
            tol,
        ).item()
    )


def _cv_square_lift_matrix(q, x):
    """
    2x2 PSD lift encoding q >= x^2 through:
      [[q, x], [x, 1]] >= 0.
    """
    return cp.vstack([cp.hstack([q, x]), cp.hstack([x, 1.0])])


def _solve_cv_df_sdp(
    protocol: CVQDB,
    solver: str = "SCS",
) -> Tuple[SDPSolveResult, List[Dict[str, float | int]], List[float]]:
    """
    Distance-fraud CV SDP (Gaussian lifted scalar form).

    Attacker chooses one forged response value per disclosed setting pair (a, b),
    where b selects the response quadrature the verifier will measure.
    The squared bias mu_{a,b}^2 is lifted via a 2x2 PSD constraint.
    """
    if cp is None:
        raise RuntimeError(
            "cvxpy is required for SDP solving. Install with: uv add cvxpy (or pip install cvxpy)."
        )

    var_m = [float(protocol.var_m(a).item()) for a in (0, 1)]
    response_vars = [float(protocol.response_measurement_var(b).item()) for b in (0, 1)]

    mu = cp.Variable((2, 2))
    q = cp.Variable((2, 2))
    constraints = []
    for a in (0, 1):
        for b in (0, 1):
            constraints.append(_cv_square_lift_matrix(q[a, b], mu[a, b]) >> 0)

    mse_expr = [[var_m[a] + q[a, b] + response_vars[b] for b in (0, 1)] for a in (0, 1)]
    prob = cp.Problem(
        cp.Minimize(
            0.25 * (mse_expr[0][0] + mse_expr[0][1] + mse_expr[1][0] + mse_expr[1][1])
        ),
        constraints,
    )
    _solve_problem(prob, solver)
    if not _cv_is_optimal_status(str(prob.status)):
        raise RuntimeError(f"CV DF SDP failed with status={prob.status}.")

    mse_vals: List[float] = []
    rows: List[Dict[str, float | int]] = []
    probs: List[float] = []
    for a in (0, 1):
        for b in (0, 1):
            mse = float(max(mse_expr[a][b].value, 1e-12))
            succ = _cv_success_prob_from_variance(mse, protocol.accept_tol)
            mse_vals.append(mse)
            probs.append(succ)
            rows.append(
                {
                    "a": int(a),
                    "b": int(b),
                    "var_m": float(var_m[a]),
                    "response_var": float(response_vars[b]),
                    "bias_mu": float(mu[a, b].value),
                    "bias_sq": float(max(q[a, b].value, 0.0)),
                    "total_error_var": mse,
                    "success": succ,
                }
            )

    return (
        SDPSolveResult(status=str(prob.status), value=float(np.mean(probs))),
        rows,
        mse_vals,
    )


def _normal_pdf_np(x: np.ndarray | float, mean: float, var: float):
    var = max(float(var), 1e-12)
    return np.exp(-0.5 * ((x - mean) ** 2) / var) / np.sqrt(2.0 * np.pi * var)


def _cv_success_prob_from_variance_array(
    var: np.ndarray,
    tol: torch.Tensor,
) -> np.ndarray:
    v = np.maximum(np.asarray(var, dtype=np.float64), 1e-12)
    scaled = float(tol.item()) / np.sqrt(2.0 * v)
    if scipy_special is not None:
        return np.asarray(scipy_special.erf(scaled), dtype=np.float64)
    return torch.erf(torch.as_tensor(scaled, dtype=TORCH_FLOAT)).detach().cpu().numpy()


def _cv_preask_stats(
    protocol: CVQDB,
    phi: float,
    probe_mean: np.ndarray,
    probe_var: np.ndarray,
    active_noise: float = 0.0,
    inactive_noise: float | None = None,
) -> dict[tuple[int, int], tuple[float, float]]:
    """
    Mean/variance of scalar pre-ask observation Z_phi conditioned on (a,b).

    probe_mean[a] and probe_var[a] describe the fake probe quadrature measured
    by the honest prover: a=0 means X_P and a=1 means P_P.
    active_noise is added to the response quadrature selected by b; when
    inactive_noise is omitted, the unselected response quadrature has vacuum
    variance hbar/2.
    """
    if inactive_noise is None:
        inactive_noise = protocol.hbar / 2.0

    w = [float(np.cos(phi)), float(np.sin(phi))]
    stats: dict[tuple[int, int], tuple[float, float]] = {}

    for a in (0, 1):
        g_a = float(protocol.honest_response_gain(a, 0).item())
        for b in (0, 1):
            active_w = w[b]
            inactive_w = w[1 - b]
            mean = active_w * g_a * float(probe_mean[a])
            var = active_w * active_w * (
                float(probe_var[a]) + active_noise
            ) + inactive_w * inactive_w * float(inactive_noise)
            stats[(a, b)] = (mean, max(var, 1e-12))

    return stats


def _cv_preask_theta_stats(
    protocol: CVQDB,
    theta_grid: np.ndarray,
) -> list[tuple[float, float, np.ndarray]]:
    stats = []
    for theta in theta_grid:
        theta_t = torch.tensor(float(theta), dtype=TORCH_FLOAT)
        stats.append(
            (
                float(theta),
                float(max(protocol.var_y(theta_t).item(), 1e-12)),
                np.array(
                    [float(protocol.cov_m_y(a, theta_t).item()) for a in (0, 1)],
                    dtype=np.float64,
                ),
            )
        )
    return stats


def _cv_mf_preask_value_for_probe(
    protocol: CVQDB,
    probe_mean: np.ndarray,
    probe_var: np.ndarray,
    phi: float,
    theta_grid: np.ndarray,
    theta_stats: list[tuple[float, float, np.ndarray]] | None = None,
    z_grid_size: int = 1001,
    z_sigma: float = 8.0,
    active_noise: float = 0.0,
    inactive_noise: float | None = None,
) -> tuple[float, dict[str, object]]:
    """
    Deterministic 1D quadrature over the pre-ask observation z.

    For each z, the challenge measurement angle is selected adaptively from
    theta_grid using the posterior Pr[a,b | z].
    """
    if z_grid_size < 2:
        raise ValueError("z_grid_size must be >= 2")
    if z_sigma <= 0.0:
        raise ValueError("z_sigma must be > 0")
    if inactive_noise is None:
        inactive_noise = protocol.hbar / 2.0

    stats = _cv_preask_stats(
        protocol=protocol,
        phi=phi,
        probe_mean=probe_mean,
        probe_var=probe_var,
        active_noise=active_noise,
        inactive_noise=inactive_noise,
    )

    z_min = min(mu - z_sigma * math.sqrt(var) for mu, var in stats.values())
    z_max = max(mu + z_sigma * math.sqrt(var) for mu, var in stats.values())
    z_grid = np.linspace(z_min, z_max, z_grid_size, dtype=np.float64)

    branch_density = np.empty((z_grid_size, 2, 2), dtype=np.float64)
    for a in (0, 1):
        for b in (0, 1):
            mean_z, var_z = stats[(a, b)]
            branch_density[:, a, b] = _normal_pdf_np(z_grid, mean_z, var_z)

    weighted_density = 0.25 * branch_density
    mixture_density = np.sum(weighted_density, axis=(1, 2))
    posterior_ab = np.divide(
        weighted_density,
        mixture_density[:, None, None],
        out=np.zeros_like(weighted_density),
        where=mixture_density[:, None, None] > 0.0,
    )

    var_m = np.array([float(protocol.var_m(a).item()) for a in (0, 1)])
    response_vars = np.array(
        [float(protocol.response_measurement_var(b).item()) for b in (0, 1)]
    )

    if theta_stats is None:
        theta_stats = _cv_preask_theta_stats(protocol, theta_grid)

    best_score_by_z = np.full(z_grid_size, -1.0, dtype=np.float64)
    best_theta_by_z = np.zeros(z_grid_size, dtype=np.float64)

    for theta, var_y, cov in theta_stats:
        gains = np.zeros((z_grid_size, 2), dtype=np.float64)
        for b in (0, 1):
            denom = posterior_ab[:, 0, b] + posterior_ab[:, 1, b]
            numer = (
                posterior_ab[:, 0, b] * cov[0] / var_y
                + posterior_ab[:, 1, b] * cov[1] / var_y
            )
            gains[:, b] = np.divide(
                numer,
                denom,
                out=np.zeros_like(numer),
                where=denom > 1e-14,
            )

        local_score = np.zeros(z_grid_size, dtype=np.float64)
        for a in (0, 1):
            for b in (0, 1):
                k_b = gains[:, b]
                err_var = (
                    var_m[a] + k_b * k_b * var_y - 2.0 * k_b * cov[a] + response_vars[b]
                )
                succ = _cv_success_prob_from_variance_array(
                    err_var, protocol.accept_tol
                )
                local_score += posterior_ab[:, a, b] * succ

        improved = local_score > best_score_by_z
        best_score_by_z[improved] = local_score[improved]
        best_theta_by_z[improved] = theta

    value = float(np.trapezoid(mixture_density * best_score_by_z, z_grid))
    dz = float(z_grid[1] - z_grid[0])
    theta_hist = np.histogram(
        np.degrees(best_theta_by_z),
        bins=18,
        range=(0.0, 360.0),
        weights=mixture_density * dz,
    )[0]

    meta = {
        "phi": float(phi),
        "probe_mean_x": float(probe_mean[0]),
        "probe_mean_p": float(probe_mean[1]),
        "probe_var_x": float(probe_var[0]),
        "probe_var_p": float(probe_var[1]),
        "active_response_quadrature_added_noise": float(active_noise),
        "inactive_response_quadrature_variance": float(inactive_noise),
        "z_grid_size": int(z_grid_size),
        "z_sigma": float(z_sigma),
        "z_grid_min": float(z_min),
        "z_grid_max": float(z_max),
        "z_integration_rule": "trapezoid",
        "adaptive_theta_weighted_histogram_18_bins": [float(x) for x in theta_hist],
    }
    return value, meta


def _solve_cv_mf_preask_grid(
    protocol: CVQDB,
    probe_amp_grid: np.ndarray | None = None,
    probe_mean_angle_grid: np.ndarray | None = None,
    probe_squeeze_grid: np.ndarray | None = None,
    phi_grid_size: int = 181,
    theta_grid_size: int = 181,
    z_grid_size: int = 1001,
    z_sigma: float = 8.0,
    max_probe_energy: float = 2.0,
) -> tuple[SDPSolveResult, dict[str, object]]:
    """
    Restricted Gaussian MF pre-ask benchmark.

    The probe is a displaced squeezed Gaussian mode with an energy cap. This is
    not a full CV optimum; it is an energy-constrained finite Gaussian grid
    benchmark with adaptive challenge measurement. It gave the CV MF estimate
    0.9138 of the paper's first version (arXiv:2607.13464v1).

    The revised paper does not use this estimate. The published protocol puts
    no cap on the probe energy, and the bright-probe attack evaluated in
    ``verification/cv_closed_forms.py`` shows that the CV MF value is at least
    0.9901.
    """
    if phi_grid_size < 1:
        raise ValueError("phi_grid_size must be >= 1")
    if theta_grid_size < 1:
        raise ValueError("theta_grid_size must be >= 1")
    if z_grid_size < 2:
        raise ValueError("z_grid_size must be >= 2")
    if z_sigma <= 0.0:
        raise ValueError("z_sigma must be > 0")

    if probe_amp_grid is None:
        probe_amp_grid = np.linspace(0.0, 3.0, 31)

    if probe_mean_angle_grid is None:
        probe_mean_angle_grid = np.array([np.pi / 4.0])

    if probe_squeeze_grid is None:
        probe_squeeze_grid = np.array([0.0])

    phi_grid_endpoint = False
    theta_grid_endpoint = False
    phi_grid = np.linspace(0.0, 2.0 * np.pi, phi_grid_size, endpoint=phi_grid_endpoint)
    theta_grid = np.linspace(
        0.0, 2.0 * np.pi, theta_grid_size, endpoint=theta_grid_endpoint
    )
    theta_stats = _cv_preask_theta_stats(protocol, theta_grid)
    v0 = protocol.hbar / 2.0
    active_noise = 0.0
    inactive_noise = v0

    best_value = -1.0
    best_meta: dict[str, object] = {}
    feasible_probes = 0
    evaluated_points = 0

    for squeeze in probe_squeeze_grid:
        probe_var = np.array(
            [
                v0 * np.exp(-2.0 * float(squeeze)),
                v0 * np.exp(+2.0 * float(squeeze)),
            ],
            dtype=np.float64,
        )

        for amp in probe_amp_grid:
            for mean_angle in probe_mean_angle_grid:
                probe_mean = float(amp) * np.array(
                    [np.cos(float(mean_angle)), np.sin(float(mean_angle))],
                    dtype=np.float64,
                )
                nbar = (
                    float(np.sum(probe_var) + np.dot(probe_mean, probe_mean))
                    / (2.0 * protocol.hbar)
                    - 0.5
                )

                if nbar > max_probe_energy:
                    continue

                feasible_probes += 1
                for phi in phi_grid:
                    value, meta = _cv_mf_preask_value_for_probe(
                        protocol=protocol,
                        probe_mean=probe_mean,
                        probe_var=probe_var,
                        phi=float(phi),
                        theta_grid=theta_grid,
                        theta_stats=theta_stats,
                        z_grid_size=z_grid_size,
                        z_sigma=z_sigma,
                        active_noise=active_noise,
                        inactive_noise=inactive_noise,
                    )
                    evaluated_points += 1

                    if value > best_value:
                        best_value = value
                        best_meta = {
                            **meta,
                            "probe_energy_nbar": float(nbar),
                            "probe_squeeze": float(squeeze),
                            "probe_amp": float(amp),
                            "probe_mean_angle": float(mean_angle),
                            "phi_grid_size": int(phi_grid_size),
                            "theta_grid_size": int(theta_grid_size),
                            "phi_grid_endpoint": bool(phi_grid_endpoint),
                            "theta_grid_endpoint": bool(theta_grid_endpoint),
                            "max_probe_energy": float(max_probe_energy),
                        }

    if best_value < 0.0:
        raise RuntimeError("CV MF pre-ask grid found no feasible probe.")

    best_meta["feasible_probe_count"] = int(feasible_probes)
    best_meta["evaluated_probe_phi_points"] = int(evaluated_points)
    return SDPSolveResult(status="grid_optimal", value=float(best_value)), best_meta


def run_cv_protocol_sdp(
    protocol: CVQDB,
    solver: str = "SCS",
    mf_preask_max_probe_energy: float = 2.0,
    mf_preask_phi_grid_size: int = 181,
    mf_preask_theta_grid_size: int = 181,
    mf_preask_z_grid_size: int = 1001,
    mf_preask_z_sigma: float = 8.0,
) -> Dict[str, object]:
    """
    CV-QDB Gaussian estimation analysis of the paper's first version.

    DF is solved via a small scalar SDP with PSD lifts for squared terms.
    MF is an energy-constrained finite Gaussian pre-ask grid benchmark with
    adaptive challenge measurements, not a continuous Gaussian optimum.

    The revised paper shows that this DF value is exact over all strategies
    and replaces the MF estimate by the bright-probe attack. Its CV-QDB values
    are printed by the ``cv-values`` subcommand and in the CV row of ``table``.
    """
    df, _df_rows, _df_mse = _solve_cv_df_sdp(protocol, solver=solver)
    mf_preask, mf_preask_meta = _solve_cv_mf_preask_grid(
        protocol,
        max_probe_energy=mf_preask_max_probe_energy,
        phi_grid_size=mf_preask_phi_grid_size,
        theta_grid_size=mf_preask_theta_grid_size,
        z_grid_size=mf_preask_z_grid_size,
        z_sigma=mf_preask_z_sigma,
    )
    completeness_status, completeness_value = _protocol_completeness(protocol)

    df_known, mf_known = _protocol_known_vars("cv")
    out: Dict[str, object] = {
        "protocol": "cv",
        "solver": solver.upper(),
        "df_known": list(df_known),
        "mf_known": list(mf_known),
        "df_status": df.status,
        "df_value": df.value,
        "mf_status": mf_preask.status,
        "mf_value": mf_preask.value,
        "mf_quantity": (
            "energy-capped Gaussian grid estimate of arXiv:2607.13464v1; "
            "not the MF value"
        ),
        "mf_preask_status": mf_preask.status,
        "mf_preask_value": mf_preask.value,
        "mf_preask_meta": mf_preask_meta,
        "completeness_status": completeness_status,
        "completeness_value": completeness_value,
        "cv_params": {
            "epr_r": float(protocol.epr_r),
            "epr_phase": float(protocol.epr_phase),
            "hbar": float(protocol.hbar),
            "accept_tol": float(protocol.accept_tol.item()),
            "response_var": float(protocol.response_var.item()),
            "mf_preask_max_probe_energy": float(mf_preask_max_probe_energy),
            "mf_preask_phi_grid_size": int(mf_preask_phi_grid_size),
            "mf_preask_phi_grid_endpoint": False,
            "mf_preask_theta_grid_size": int(mf_preask_theta_grid_size),
            "mf_preask_theta_grid_endpoint": False,
            "mf_preask_z_grid_size": int(mf_preask_z_grid_size),
            "mf_preask_z_sigma": float(mf_preask_z_sigma),
            "mf_preask_active_response_quadrature_added_noise": 0.0,
            "mf_preask_inactive_response_quadrature_variance": float(
                protocol.hbar / 2.0
            ),
        },
    }
    return out


def run_protocol_sdp(
    protocol: str,
    solver: str = "SCS",
    cv_params: Dict[str, float | int | None] | None = None,
) -> Dict[str, object]:
    p = _canonical_protocol_key(protocol)
    if p == "cv":
        params = {} if cv_params is None else dict(cv_params)
        response_var_raw = params.get("response_var")
        response_var = None if response_var_raw is None else float(response_var_raw)
        cv_protocol = CVQDB(
            epr_r=float(params.get("epr_r", 1.2)),
            accept_tol=float(params.get("accept_tol", 1.1)),
            response_var=response_var,
        )
        return run_cv_protocol_sdp(
            protocol=cv_protocol,
            solver=solver,
            mf_preask_max_probe_energy=float(
                params.get("mf_preask_max_probe_energy", 2.0)
            ),
            mf_preask_phi_grid_size=int(params.get("mf_preask_phi_grid_size", 181)),
            mf_preask_theta_grid_size=int(params.get("mf_preask_theta_grid_size", 181)),
            mf_preask_z_grid_size=int(params.get("mf_preask_z_grid_size", 1001)),
            mf_preask_z_sigma=float(params.get("mf_preask_z_sigma", 8.0)),
        )

    protocol_obj = _dv_protocol_instance(protocol)
    rounds = build_round_entries(protocol)
    df_known, mf_known = _protocol_known_vars(protocol)
    completeness_status, completeness_value = _protocol_completeness(protocol_obj)

    df = solve_df_state_sdp(
        rounds,
        known_vars=df_known,
        solver=solver,
    )
    mf = solve_mf_full_comb_sdp(
        protocol=protocol,
        solver=solver,
    )

    out: Dict[str, object] = {
        "protocol": p,
        "solver": solver.upper(),
        "df_known": list(df_known),
        "mf_known": list(mf_known),
        "df_status": df.status,
        "df_value": df.value,
        "mf_status": mf.status,
        "mf_value": mf.value,
        "completeness_status": completeness_status,
        "completeness_value": completeness_value,
    }

    session_summary = protocol_obj.get_honest_session_summary()
    if session_summary is not None:
        out["session_summary"] = session_summary
        if p == "e91":
            out["e91_authentication"] = session_summary
        if p == "qdb2017":
            out["qdb2017_verification"] = session_summary
        if p == "mutual-entanglement":
            out["mutual_entanglement_session"] = session_summary

    return out


# === cli.py ===

"""Command-line interface for reproducing QDB benchmark values."""


def run_certificate_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a reduced one-round MF comb SDP numerical witness and "
            "check its floating-point residuals."
        )
    )
    parser.add_argument(
        "--protocol",
        default="qdb2019",
        help=(
            "DV protocol key. Supported: qdb2017, qdb2019, "
            "mutual-entanglement, e91. Default: qdb2019."
        ),
    )
    parser.add_argument(
        "--solver",
        default="CLARABEL",
        help="CVXPY solver used to produce the witness. Default: CLARABEL.",
    )
    parser.add_argument(
        "--output",
        "--path",
        dest="path",
        default=None,
        help=(
            "Numerical witness .npz path. Default: "
            "<repo-root>/certificates/<protocol>_mf_certificate.npz."
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="Numerical tolerance for the NumPy checker. Default: 1e-8.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check an existing witness instead of solving and saving first.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON only.",
    )
    args = parser.parse_args(argv)

    try:
        protocol_key = _canonical_protocol_key(str(args.protocol))
        if protocol_key == "cv":
            raise ValueError("MF witnesses are implemented for finite DV SDPs only.")
        _dv_protocol_instance(protocol_key)
    except ValueError as e:
        raise SystemExit(str(e))

    cert_path = (
        _default_mf_certificate_path(protocol_key)
        if args.path is None
        else _resolve_certificate_path(args.path)
    )

    saved: Dict[str, object] | None = None
    if not args.check_only:
        try:
            saved = save_mf_certificate(
                protocol=protocol_key,
                path=cert_path,
                solver=args.solver,
                compressed=not args.no_compress,
            )
        except RuntimeError as e:
            raise SystemExit(str(e))
        except ValueError as e:
            raise SystemExit(str(e))

    try:
        checked = check_mf_certificate(
            protocol=protocol_key,
            path=cert_path,
            tol=float(args.tol),
            verbose=not args.json,
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    except KeyError as e:
        raise SystemExit(f"Numerical witness is missing required array: {e}")
    except ValueError as e:
        raise SystemExit(str(e))

    if args.json:
        payload = {"saved": saved, "checked": checked}
        print(json.dumps(payload, indent=2))
        return

    if saved is not None:
        print()
        print(f"Saved numerical witness to: {saved['path']}")


def run_exact_certificate_cli(argv: list[str] | None = None) -> None:
    """Run the solver-free exact DV certificate verifiers."""
    try:
        from verification.verify_all import main as verify_exact
    except ImportError as error:
        raise SystemExit(
            "Exact verifiers are unavailable. Run this command from the source "
            "artifact or install the verification package."
        ) from error
    verify_exact(argv)


def _load_cv_closed_forms():
    """Import the closed-form CV-QDB module shipped next to the exact verifiers."""
    try:
        from verification import cv_closed_forms
    except ImportError as error:
        raise SystemExit(
            "The CV-QDB closed forms are unavailable. Run this command from the "
            "source artifact or install the verification package."
        ) from error
    return cv_closed_forms


def run_cv_values_cli(argv: list[str] | None = None) -> None:
    """Print the closed-form CV-QDB values of the paper."""
    _load_cv_closed_forms().main(argv)


def run_sdp_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Integrated analysis: SDP for QDB2017/QDB2019/"
            "mutual-entanglement/e91 and, for CV-QDB, the Gaussian estimates of "
            "the paper's first version (arXiv:2607.13464v1). The CV-QDB values "
            "of the revised paper are printed by the cv-values subcommand."
        )
    )
    parser.add_argument(
        "--protocol",
        default="qdb2017",
        help=(
            "Protocol key to analyze. Supported: qdb2017, qdb2019, "
            "mutual-entanglement, e91, cv."
        ),
    )
    parser.add_argument(
        "--solver",
        default="SCS",
        help="CVXPY solver (e.g., SCS, MOSEK). Default: SCS.",
    )
    parser.add_argument(
        "--cv-epr-r",
        type=float,
        default=1.2,
        help="CV only: TMSV squeezing parameter r. Default: 1.2.",
    )
    parser.add_argument(
        "--cv-accept-tol",
        type=float,
        default=1.1,
        help="CV only: acceptance threshold for |m-m''|. Default: 1.1.",
    )
    parser.add_argument(
        "--cv-response-var",
        type=float,
        default=None,
        help=(
            "CV only: verifier measurement noise variance on response mode C. "
            "Default is 0 (noiseless)."
        ),
    )
    parser.add_argument(
        "--cv-mf-preask-energy",
        type=float,
        default=2.0,
        help="CV only: max pre-ask probe energy N_pre. Default: 2.0.",
    )
    parser.add_argument(
        "--cv-mf-preask-phi-grid",
        type=int,
        default=181,
        help="CV only: grid size for pre-ask measurement angle phi. Default: 181.",
    )
    parser.add_argument(
        "--cv-mf-preask-theta-grid",
        type=int,
        default=181,
        help="CV only: grid size for adaptive challenge angle theta. Default: 181.",
    )
    parser.add_argument(
        "--cv-mf-preask-z-grid",
        type=int,
        default=1001,
        help="CV only: grid size for pre-ask observation quadrature. Default: 1001.",
    )
    parser.add_argument(
        "--cv-mf-preask-z-sigma",
        type=float,
        default=8.0,
        help=(
            "CV only: branch standard-deviation envelope for the pre-ask "
            "observation quadrature grid. Default: 8.0."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON only.",
    )
    args = parser.parse_args(argv)

    try:
        protocol_key = _canonical_protocol_key(str(args.protocol))
    except ValueError as e:
        raise SystemExit(str(e))

    cv_params: Dict[str, float | int | None] | None = None
    if protocol_key == "cv":
        cv_params = {
            "epr_r": float(args.cv_epr_r),
            "accept_tol": float(args.cv_accept_tol),
            "response_var": args.cv_response_var,
            "mf_preask_max_probe_energy": float(args.cv_mf_preask_energy),
            "mf_preask_phi_grid_size": int(args.cv_mf_preask_phi_grid),
            "mf_preask_theta_grid_size": int(args.cv_mf_preask_theta_grid),
            "mf_preask_z_grid_size": int(args.cv_mf_preask_z_grid),
            "mf_preask_z_sigma": float(args.cv_mf_preask_z_sigma),
        }

    try:
        out = run_protocol_sdp(
            protocol=protocol_key,
            solver=args.solver,
            cv_params=cv_params,
        )
    except RuntimeError as e:
        raise SystemExit(str(e))
    except ValueError as e:
        raise SystemExit(str(e))

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(f"=== {protocol_key.upper()} Analysis ===")
    print(f"Solver: {out['solver']}")
    if protocol_key == "cv":
        print(
            "Gaussian estimates of the paper's first version "
            "(arXiv:2607.13464v1). For the CV-QDB values of the revised paper, "
            "run the cv-values subcommand."
        )
        print(
            "DF  (no-online-response Gaussian benchmark): "
            f"status={out['df_status']} value={out['df_value']:.6f}"
        )
        print(
            "MF  (pre-ask Gaussian grid with a probe-energy cap; not the MF value): "
            f"status={out['mf_status']} value={out['mf_value']:.6f}"
        )
        meta = out.get("mf_preask_meta")
        if isinstance(meta, dict):
            z_min = meta.get("z_grid_min")
            z_max = meta.get("z_grid_max")
            z_sigma = meta.get("z_sigma")
            z_grid_size = meta.get("z_grid_size")
            integration_rule = meta.get("z_integration_rule")
            if all(
                isinstance(x, (int, float))
                for x in (z_min, z_max, z_sigma, z_grid_size)
            ) and isinstance(integration_rule, str):
                print(
                    "MF  Z_phi grid: "
                    f"size={int(z_grid_size)} z_sigma={float(z_sigma):.1f} "
                    f"range=[{float(z_min):.6g}, {float(z_max):.6g}] "
                    f"integration={integration_rule}"
                )
    else:
        print(
            "DF  (no-online-response state SDP): "
            f"status={out['df_status']} value={out['df_value']:.6f}"
        )
        print(
            "MF  (full lifted comb SDP): "
            f"status={out['mf_status']} value={out['mf_value']:.6f}"
        )
    print("TF: not reported; outside the reduced one-round online model.")
    print(
        "Completeness (honest run): "
        f"status={out['completeness_status']} value={out['completeness_value']:.6f}"
    )
    if protocol_key == "qdb2017":
        verification_obj = out.get("qdb2017_verification")
        if isinstance(verification_obj, dict):
            rapid_raw = verification_obj.get("rapid_phase_success")
            if isinstance(rapid_raw, (int, float)):
                print(f"QDB2017 rapid-phase success (honest): {float(rapid_raw):.6f}")
            mac_raw = verification_obj.get("final_mac_success")
            if isinstance(mac_raw, (int, float)):
                print(f"QDB2017 final MAC success (honest): {float(mac_raw):.6f}")
            attack_scope = verification_obj.get("attack_scope")
            if isinstance(attack_scope, str):
                print(f"QDB2017 attack scope: {attack_scope}")
    if protocol_key == "mutual-entanglement":
        session_obj = out.get("mutual_entanglement_session")
        if isinstance(session_obj, dict):
            b_to_a_raw = session_obj.get("b_to_a_success")
            if isinstance(b_to_a_raw, (int, float)):
                print(f"Mutual B->A success (honest): {float(b_to_a_raw):.6f}")
            a_to_b_raw = session_obj.get("a_to_b_success")
            if isinstance(a_to_b_raw, (int, float)):
                print(f"Mutual A->B success (honest): {float(a_to_b_raw):.6f}")
            open_raw = session_obj.get("commitment_open_success")
            if isinstance(open_raw, (int, float)):
                print(f"Mutual opening success (honest): {float(open_raw):.6f}")
            attack_scope = session_obj.get("attack_scope")
            if isinstance(attack_scope, str):
                print(f"Mutual attack scope: {attack_scope}")
    if protocol_key == "e91":
        auth_obj = out.get("e91_authentication")
        if isinstance(auth_obj, dict):
            counts = auth_obj.get("round_type_counts")
            if isinstance(counts, dict):
                print(f"E91 round types: {counts}")
            chsh_raw = auth_obj.get("chsh_magnitude", auth_obj.get("chsh_value"))
            if isinstance(chsh_raw, (int, float)):
                print(f"E91 CHSH magnitude (honest): {float(chsh_raw):.6f}")
            attack_scope = auth_obj.get("attack_scope")
            if isinstance(attack_scope, str):
                print(f"E91 attack scope: {attack_scope}")


def run_benchmark_cli(argv: List[str] | None = None):
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0].lower() in {"certificate", "cert"}:
        return run_certificate_cli(args[1:])
    if args and args[0].lower() in {
        "exact-certificate",
        "exact-cert",
        "verify-exact",
    }:
        return run_exact_certificate_cli(args[1:])
    if args and args[0].lower() in {"cv-values", "cv_values"}:
        return run_cv_values_cli(args[1:])
    if args and args[0].lower() == "sdp":
        return run_sdp_cli(args[1:])
    return run_sdp_cli(args)


# === scripts/reproduce_table.py ===

"""Print the reduced one-round benchmark values used in the paper table."""


TABLE_PROTOCOLS = ["qdb2017", "qdb2019", "mutual-entanglement", "e91", "cv"]


def _cv_table_entry() -> Dict[str, object]:
    """
    CV-QDB row of the paper table.

    The DF value and the honest pass probability are closed forms. The MF
    value is the success probability of the bright-probe attack, so it bounds
    the optimum from below. The paper table lists it as at least
    ``cv_values["mafia_fraud_table_entry"]``.
    """
    values = _load_cv_closed_forms().compute_values()
    df_known, mf_known = _protocol_known_vars("cv")
    return {
        "protocol": "cv",
        "df_known": list(df_known),
        "mf_known": list(mf_known),
        "df_status": "closed_form",
        "df_value": values["distance_fraud_value"],
        "mf_status": "attack_lower_bound",
        "mf_value": values["attack_success_probability"],
        "completeness_status": "closed_form",
        "completeness_value": values["honest_pass_probability"],
        "cv_values": values,
    }


def run_table_cli(argv: List[str] | None = None) -> None:
    if argv:
        raise SystemExit("table does not accept arguments")
    results = {}
    for protocol in TABLE_PROTOCOLS:
        if protocol == "cv":
            results[protocol] = _cv_table_entry()
        else:
            results[protocol] = run_protocol_sdp(protocol, solver="SCS")
    print(json.dumps(results, indent=2))


# === scripts/cv_calibration_sweep.py ===

"""Run and plot the CV-QDB calibration sweep of the paper's first version.

The sweep uses the restricted Gaussian grid estimator and produced Figure 2 of
arXiv:2607.13464v1. The revised paper has no such figure.
"""


BASE_R = 1.2
BASE_DELTA = 1.1
BASE_RESPONSE_VAR = 0.0

PAPER_GRID = {
    "mf_preask_max_probe_energy": 2.0,
    "mf_preask_phi_grid_size": 181,
    "mf_preask_theta_grid_size": 181,
    "mf_preask_z_grid_size": 1001,
    "mf_preask_z_sigma": 8.0,
}

QUICK_GRID = {
    "mf_preask_max_probe_energy": 2.0,
    "mf_preask_phi_grid_size": 61,
    "mf_preask_theta_grid_size": 61,
    "mf_preask_z_grid_size": 401,
    "mf_preask_z_sigma": 8.0,
}


def _sweep_specs() -> list[dict[str, float | str]]:
    specs: list[dict[str, float | str]] = [
        {
            "panel": "Threshold",
            "x_name": "delta_cv",
            "x_value": delta,
            "epr_r": BASE_R,
            "accept_tol": delta,
            "response_var": BASE_RESPONSE_VAR,
        }
        for delta in (0.8, 1.1, 1.59, 2.0)
    ]
    specs.extend(
        {
            "panel": "Squeezing",
            "x_name": "r_sq",
            "x_value": r,
            "epr_r": r,
            "accept_tol": BASE_DELTA,
            "response_var": BASE_RESPONSE_VAR,
        }
        for r in (0.8, 1.0, 1.2, 1.4)
    )
    return specs


def _unique_parameter_points(
    specs: list[dict[str, float | str]],
) -> list[dict[str, float]]:
    points: dict[tuple[float, float, float], dict[str, float]] = {}
    for spec in specs:
        key = (
            float(spec["epr_r"]),
            float(spec["accept_tol"]),
            float(spec["response_var"]),
        )
        points[key] = {
            "epr_r": key[0],
            "accept_tol": key[1],
            "response_var": key[2],
        }
    return list(points.values())


def _run_point(job: dict[str, Any]) -> dict[str, Any]:
    point = dict(job["point"])
    grid = dict(job["grid"])
    solver = str(job["solver"])
    cv_params = {**point, **grid}
    out = run_protocol_sdp("cv", solver=solver, cv_params=cv_params)
    meta = out["mf_preask_meta"]
    return {
        **point,
        "df": float(out["df_value"]),
        "mf": float(out["mf_value"]),
        "honest": float(out["completeness_value"]),
        "df_status": str(out["df_status"]),
        "mf_status": str(out["mf_status"]),
        "completeness_status": str(out["completeness_status"]),
        "best_phi": float(meta["phi"]),
        "probe_amp": float(meta["probe_amp"]),
        "probe_energy_nbar": float(meta["probe_energy_nbar"]),
    }


def _run_sweep(
    specs: list[dict[str, float | str]],
    grid: dict[str, float | int],
    solver: str,
    workers: int,
) -> list[dict[str, Any]]:
    points = _unique_parameter_points(specs)
    jobs = [{"point": point, "grid": grid, "solver": solver} for point in points]
    results: list[dict[str, Any]] = []
    if workers == 1:
        for job in jobs:
            result = _run_point(job)
            print(_format_progress(result), flush=True)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_point, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                print(_format_progress(result), flush=True)
                results.append(result)

    results.sort(key=lambda row: (row["epr_r"], row["accept_tol"], row["response_var"]))
    return results


def _format_progress(row: dict[str, Any]) -> str:
    return (
        "completed "
        f"r={row['epr_r']:.3g} delta={row['accept_tol']:.3g} "
        f"response_var={row['response_var']:.3g}: "
        f"honest={row['honest']:.6f} DF={row['df']:.6f} MF={row['mf']:.6f}"
    )


def _join_results_to_specs(
    specs: list[dict[str, float | str]],
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    _require_cv_sweep_dependencies()
    by_key = {
        (
            float(row["epr_r"]),
            float(row["accept_tol"]),
            float(row["response_var"]),
        ): row
        for row in results
    }
    rows: list[dict[str, Any]] = []
    for spec in specs:
        key = (
            float(spec["epr_r"]),
            float(spec["accept_tol"]),
            float(spec["response_var"]),
        )
        result = by_key[key]
        for metric, value in (
            ("Completeness", result["honest"]),
            ("Distance-fraud", result["df"]),
            ("Mafia-fraud", result["mf"]),
        ):
            rows.append(
                {
                    **spec,
                    "metric": metric,
                    "value": float(value),
                    "df_status": result["df_status"],
                    "mf_status": result["mf_status"],
                    "completeness_status": result["completeness_status"],
                    "best_phi": result["best_phi"],
                    "probe_amp": result["probe_amp"],
                    "probe_energy_nbar": result["probe_energy_nbar"],
                }
            )
    return pd.DataFrame(rows)


def _plot_sweep(long_df: pd.DataFrame, path: Path) -> list[Path]:
    _require_cv_sweep_dependencies()
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    palette = {
        "Completeness": "#167f5f",
        "Distance-fraud": "#b33a3a",
        "Mafia-fraud": "#2d5aa7",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15), sharey=True)

    panel_specs = [
        (
            "Threshold",
            r"$\delta_{\mathrm{cv}}$",
            r"$r_{\mathrm{sq}}=1.2,\ \sigma_{\mathrm{resp}}^2=0$",
            "(a)",
            (0.98, 0.06, "right"),
        ),
        (
            "Squeezing",
            r"$r_{\mathrm{sq}}$",
            r"$\delta_{\mathrm{cv}}=1.1,\ \sigma_{\mathrm{resp}}^2=0$",
            "(b)",
            (0.06, 0.06, "left"),
        ),
    ]

    for ax, (panel, xlabel, annotation, panel_label, annotation_pos) in zip(
        axes, panel_specs
    ):
        panel_df = long_df[long_df["panel"] == panel].copy()
        sns.lineplot(
            data=panel_df,
            x="x_value",
            y="value",
            hue="metric",
            style="metric",
            markers=True,
            dashes=False,
            linewidth=2.2,
            markersize=6.5,
            palette=palette,
            ax=ax,
            legend=panel == "Threshold",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("One-round success probability" if panel == "Threshold" else "")
        ax.set_ylim(0.2, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#d9d9d9", linewidth=0.7)

        paper_x = BASE_DELTA if panel == "Threshold" else BASE_R
        paper_rows = panel_df[np.isclose(panel_df["x_value"].astype(float), paper_x)]
        ax.axvline(
            paper_x,
            color="#202020",
            linestyle=(0, (2.5, 2.5)),
            linewidth=1.0,
            alpha=0.65,
            zorder=1,
        )
        ax.scatter(
            paper_rows["x_value"],
            paper_rows["value"],
            s=78,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.45,
            zorder=6,
        )
        ax.text(
            0.015,
            0.995,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            annotation_pos[0],
            annotation_pos[1],
            annotation,
            transform=ax.transAxes,
            ha=annotation_pos[2],
            va="bottom",
            fontsize=8.5,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.095),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        bottom=0.31,
        top=0.95,
        wspace=0.07,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    written_paths = [path]
    if path.suffix.lower() != ".pdf":
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path)
        written_paths.append(pdf_path)
    plt.close(fig)
    return written_paths


def parse_cv_sweep_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CV-QDB calibration sweep of the paper's first version "
            "(Figure 2 of arXiv:2607.13464v1) and render a seaborn/matplotlib "
            "plot. The revised paper does not use this sweep."
        )
    )
    parser.add_argument(
        "--grid-profile",
        choices=("paper", "quick"),
        default="paper",
        help=(
            "Use the grid of the paper's first version or a smaller exploratory "
            "grid. Default: paper."
        ),
    )
    parser.add_argument("--solver", default="SCS", help="CVXPY solver. Default: SCS.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(3, os.cpu_count() or 1),
        help="Number of parallel parameter-point workers. Default: min(3, CPU count).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV and JSON outputs. Default: results.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for plot output. Default: figures.",
    )
    parser.add_argument(
        "--figure-name",
        default="cv_calibration_sweep.png",
        help="Figure filename. Default: cv_calibration_sweep.png.",
    )
    return parser.parse_args(argv)


def run_cv_sweep_cli(argv: List[str] | None = None) -> None:
    args = parse_cv_sweep_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    print(
        "Calibration sweep of the paper's first version (arXiv:2607.13464v1). "
        "The revised paper does not use it.",
        flush=True,
    )
    grid = PAPER_GRID if args.grid_profile == "paper" else QUICK_GRID
    specs = _sweep_specs()
    results = _run_sweep(specs, grid=grid, solver=args.solver, workers=args.workers)
    long_df = _join_results_to_specs(specs, results)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / "cv_calibration_sweep.csv"
    json_path = args.results_dir / "cv_calibration_sweep.json"
    figure_path = args.figure_dir / args.figure_name

    long_df.to_csv(csv_path, index=False)
    payload = {
        "grid_profile": args.grid_profile,
        "solver": args.solver,
        "grid": grid,
        "parameter_results": results,
        "panel_rows": long_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    figure_paths = _plot_sweep(long_df, figure_path)

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    for written_path in figure_paths:
        print(f"wrote {written_path}")


# === standalone dispatch ===


def main(argv: List[str] | None = None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0].lower() in {"table", "reproduce-table", "reproduce_table"}:
        return run_table_cli(args[1:])
    if args and args[0].lower() in {"cv-sweep", "cv_calibration_sweep", "sweep"}:
        return run_cv_sweep_cli(args[1:])
    return run_benchmark_cli(args)


if __name__ == "__main__":
    main()
