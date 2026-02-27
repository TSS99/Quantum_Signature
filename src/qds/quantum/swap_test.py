from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qds.quantum.utils import ensure_statevector, statevector_to_circuit


def build_swap_test_circuit(state_a_circuit: QuantumCircuit, state_b_circuit: QuantumCircuit) -> QuantumCircuit:
    if state_a_circuit.num_qubits != state_b_circuit.num_qubits:
        raise ValueError("Swap test requires states with the same number of qubits.")
    n = state_a_circuit.num_qubits
    anc = 0
    reg_a = list(range(1, 1 + n))
    reg_b = list(range(1 + n, 1 + 2 * n))

    qc = QuantumCircuit(1 + 2 * n, 1)
    qc.compose(state_a_circuit, qubits=reg_a, inplace=True)
    qc.compose(state_b_circuit, qubits=reg_b, inplace=True)
    qc.h(anc)
    for i in range(n):
        qc.cswap(anc, reg_a[i], reg_b[i])
    qc.h(anc)
    qc.measure(anc, 0)
    return qc


def swap_test_pass_probability_exact(
    state_a: Statevector | QuantumCircuit, state_b: Statevector | QuantumCircuit
) -> float:
    s_a = ensure_statevector(state_a)
    s_b = ensure_statevector(state_b)
    if s_a.dim != s_b.dim:
        raise ValueError("Swap test requires equal state dimensions.")
    overlap = abs(np.vdot(s_a.data, s_b.data))
    return float((1.0 + overlap * overlap) / 2.0)


def swap_test_pass_probability_sampled(
    state_a: Statevector | QuantumCircuit,
    state_b: Statevector | QuantumCircuit,
    *,
    shots: int = 1024,
    seed: int | None = None,
    noise_model=None,
) -> Tuple[float, Dict[str, int]]:
    if shots <= 0:
        raise ValueError("shots must be positive.")

    prep_a = state_a.copy() if isinstance(state_a, QuantumCircuit) else statevector_to_circuit(state_a)
    prep_b = state_b.copy() if isinstance(state_b, QuantumCircuit) else statevector_to_circuit(state_b)

    qc = build_swap_test_circuit(prep_a, prep_b)
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts(0)
    pass_prob = counts.get("0", 0) / shots
    return pass_prob, counts


def swap_test_single_shot(
    state_a: Statevector | QuantumCircuit,
    state_b: Statevector | QuantumCircuit,
    *,
    seed: int | None = None,
    noise_model=None,
) -> bool:
    pass_prob, _ = swap_test_pass_probability_sampled(
        state_a, state_b, shots=1, seed=seed, noise_model=noise_model
    )
    return pass_prob >= 1.0
