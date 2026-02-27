from __future__ import annotations

import hashlib
from typing import Final

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal
from qiskit.quantum_info import Statevector

from qds.qowf.base import QuantumOneWayFunction

SUPPORTED_N: Final[set[int]] = {3, 4, 5}


class FingerprintPhaseFunction(QuantumOneWayFunction):
    """Family 2: phase fingerprint states on n in {3,4,5} qubits."""

    family_name = "fingerprint"

    def __init__(self, *, L: int, n: int) -> None:
        if n not in SUPPORTED_N:
            raise ValueError(f"Fingerprint family supports n in {sorted(SUPPORTED_N)}, got n={n}.")
        super().__init__(L=L, n=n)
        self.N = 2**self.n

    def _expanded_hash_bits(self, k_norm: str, *, size: int) -> np.ndarray:
        key_bytes = k_norm.encode("ascii")
        bit_chunks: list[np.ndarray] = []
        current_bits = 0
        counter = 0
        while current_bits < size:
            ctr = counter.to_bytes(4, "big", signed=False)
            digest = hashlib.sha256(key_bytes + ctr).digest()
            bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
            bit_chunks.append(bits)
            current_bits += bits.size
            counter += 1
        return np.concatenate(bit_chunks)[:size]

    def phase_pattern(self, k_norm: str) -> np.ndarray:
        bits = self._expanded_hash_bits(self.normalize_key(k_norm), size=self.N)
        return bits.astype(np.uint8)

    def _statevector_from_key(self, k_norm: str) -> Statevector:
        phase_bits = self.phase_pattern(k_norm)
        signs = np.where(phase_bits == 0, 1.0, -1.0).astype(np.complex128)
        data = signs / np.sqrt(self.N)
        return Statevector(data)

    def _prepare_circuit_from_key(self, k_norm: str) -> QuantumCircuit:
        phase_bits = self.phase_pattern(k_norm)
        diag_entries = np.where(phase_bits == 0, 1.0, -1.0).astype(np.complex128)

        qc = QuantumCircuit(self.n, name=f"{self.family_name}_{k_norm}")
        qc.h(range(self.n))
        qc.append(Diagonal(diag_entries), range(self.n))
        return qc
