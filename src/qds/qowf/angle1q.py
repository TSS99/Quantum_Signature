from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qds.qowf.base import QuantumOneWayFunction


class Angle1QFunction(QuantumOneWayFunction):
    """Family 1: |f_k> = cos(j*theta)|0> + sin(j*theta)|1> on one qubit."""

    family_name = "angle1q"

    def __init__(self, *, L: int) -> None:
        super().__init__(L=L, n=1)
        self.theta = math.pi / (2**self.L)

    def _key_to_j(self, k_norm: str) -> int:
        return int(k_norm, 2)

    def _angle(self, k_norm: str) -> float:
        return self._key_to_j(k_norm) * self.theta

    def _prepare_circuit_from_key(self, k_norm: str) -> QuantumCircuit:
        angle = self._angle(k_norm)
        qc = QuantumCircuit(1, name=f"{self.family_name}_{k_norm}")
        qc.ry(2.0 * angle, 0)
        return qc

    def _statevector_from_key(self, k_norm: str) -> Statevector:
        angle = self._angle(k_norm)
        data = np.array([math.cos(angle), math.sin(angle)], dtype=np.complex128)
        return Statevector(data)
