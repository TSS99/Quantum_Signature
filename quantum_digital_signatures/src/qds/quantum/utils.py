from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


def seeded_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def ensure_statevector(obj: Statevector | QuantumCircuit) -> Statevector:
    if isinstance(obj, Statevector):
        return obj.copy()
    if isinstance(obj, QuantumCircuit):
        return Statevector.from_instruction(obj)
    raise TypeError(f"Expected Statevector or QuantumCircuit, got {type(obj)!r}.")


def statevector_to_circuit(state: Statevector) -> QuantumCircuit:
    n = int(np.log2(len(state.data)))
    qc = QuantumCircuit(n)
    qc.initialize(state.data, range(n))
    return qc


def ensure_artifacts_dir(path: str | Path = "artifacts") -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_noise_model(
    *,
    depolarizing_prob: float = 0.0,
    readout_error_prob: float = 0.0,
) -> Optional[NoiseModel]:
    if depolarizing_prob < 0 or depolarizing_prob >= 1:
        raise ValueError("depolarizing_prob must satisfy 0 <= p < 1.")
    if readout_error_prob < 0 or readout_error_prob >= 0.5:
        raise ValueError("readout_error_prob must satisfy 0 <= p < 0.5.")

    if depolarizing_prob == 0 and readout_error_prob == 0:
        return None

    noise_model = NoiseModel()
    if depolarizing_prob > 0:
        err_1q = depolarizing_error(depolarizing_prob, 1)
        for gate in ["h", "x", "sx", "ry", "rz", "u", "u1", "u2", "u3", "initialize"]:
            try:
                noise_model.add_all_qubit_quantum_error(err_1q, [gate])
            except Exception:
                continue
    if readout_error_prob > 0:
        r = readout_error_prob
        readout = ReadoutError([[1 - r, r], [r, 1 - r]])
        noise_model.add_all_qubit_readout_error(readout)
    return noise_model
