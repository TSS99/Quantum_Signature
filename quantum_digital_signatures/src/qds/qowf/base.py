from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

KeyInput = Union[str, int]


class QuantumOneWayFunction(ABC):
    """Base class for toy quantum one-way function families."""

    family_name: str

    def __init__(self, *, L: int, n: int) -> None:
        if L <= 0:
            raise ValueError("L must be positive.")
        if n <= 0:
            raise ValueError("n must be positive.")
        self.L = L
        self.n = n
        self._state_cache: Dict[str, Statevector] = {}
        self._circuit_cache: Dict[str, QuantumCircuit] = {}

    def normalize_key(self, k: KeyInput) -> str:
        """Return a canonical L-bit key string."""
        if isinstance(k, int):
            if k < 0 or k >= 2**self.L:
                raise ValueError(f"Integer key must lie in [0, {2**self.L - 1}] for L={self.L}.")
            return format(k, f"0{self.L}b")
        if not isinstance(k, str):
            raise TypeError("Key must be an int or an L-bit string.")
        if len(k) != self.L:
            raise ValueError(f"Key length {len(k)} does not match L={self.L}.")
        if any(ch not in {"0", "1"} for ch in k):
            raise ValueError("Key string must contain only '0' and '1'.")
        return k

    def prepare_circuit(self, k: KeyInput) -> QuantumCircuit:
        """Prepare |f_k> as a quantum circuit."""
        k_norm = self.normalize_key(k)
        if k_norm not in self._circuit_cache:
            self._circuit_cache[k_norm] = self._prepare_circuit_from_key(k_norm)
        return self._circuit_cache[k_norm].copy()

    def statevector(self, k: KeyInput) -> Statevector:
        """Return |f_k> as a cached statevector."""
        k_norm = self.normalize_key(k)
        if k_norm not in self._state_cache:
            self._state_cache[k_norm] = self._statevector_from_key(k_norm)
        return self._state_cache[k_norm].copy()

    def overlap(self, k1: KeyInput, k2: KeyInput) -> float:
        """Absolute inner product |<f_k1|f_k2>|."""
        s1 = self.statevector(k1)
        s2 = self.statevector(k2)
        return float(abs(np.vdot(s1.data, s2.data)))

    @abstractmethod
    def _prepare_circuit_from_key(self, k_norm: str) -> QuantumCircuit:
        """Build the circuit for a normalized key."""

    @abstractmethod
    def _statevector_from_key(self, k_norm: str) -> Statevector:
        """Build the statevector for a normalized key."""
