from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qds.qowf.base import KeyInput, QuantumOneWayFunction
from qds.quantum.utils import ensure_statevector


def verify_statevector(
    psi: Statevector | QuantumCircuit, k: KeyInput, qowf: QuantumOneWayFunction
) -> float:
    psi_sv = ensure_statevector(psi)
    target = qowf.statevector(k)
    if psi_sv.dim != target.dim:
        raise ValueError("State and target key state have different dimensions.")
    return float(abs(np.vdot(psi_sv.data, target.data)) ** 2)


def build_verification_circuit(
    psi_preparation_circuit: QuantumCircuit, k: KeyInput, qowf: QuantumOneWayFunction
) -> QuantumCircuit:
    if psi_preparation_circuit.num_qubits != qowf.n:
        raise ValueError(
            f"psi_preparation_circuit has {psi_preparation_circuit.num_qubits} qubits, expected {qowf.n}."
        )
    qc = QuantumCircuit(qowf.n, qowf.n)
    qc.compose(psi_preparation_circuit, qubits=range(qowf.n), inplace=True)
    qc.compose(qowf.prepare_circuit(k).inverse(), qubits=range(qowf.n), inplace=True)
    qc.measure(range(qowf.n), range(qowf.n))
    return qc


def verify_by_circuit(
    psi_preparation_circuit: QuantumCircuit,
    k: KeyInput,
    qowf: QuantumOneWayFunction,
    *,
    shots: int | None = None,
    seed: int | None = None,
    noise_model=None,
) -> float:
    if psi_preparation_circuit.num_qubits != qowf.n:
        raise ValueError(
            f"psi_preparation_circuit has {psi_preparation_circuit.num_qubits} qubits, expected {qowf.n}."
        )

    if shots is None:
        qc = QuantumCircuit(qowf.n)
        qc.compose(psi_preparation_circuit, qubits=range(qowf.n), inplace=True)
        qc.compose(qowf.prepare_circuit(k).inverse(), qubits=range(qowf.n), inplace=True)
        state = Statevector.from_instruction(qc)
        return float(abs(state.data[0]) ** 2)

    if shots <= 0:
        raise ValueError("shots must be positive when provided.")
    measured = build_verification_circuit(psi_preparation_circuit, k, qowf)
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    result = sim.run(measured, shots=shots).result()
    counts = result.get_counts(0)
    all_zero = "0" * qowf.n
    return counts.get(all_zero, 0) / shots
