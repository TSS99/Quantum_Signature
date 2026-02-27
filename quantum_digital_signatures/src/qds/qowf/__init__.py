from __future__ import annotations

from qds.qowf.angle1q import Angle1QFunction
from qds.qowf.base import QuantumOneWayFunction
from qds.qowf.fingerprint import FingerprintPhaseFunction


def build_qowf(*, family: str, L: int, n: int | None = None) -> QuantumOneWayFunction:
    family_norm = family.lower().strip()
    if family_norm == "angle1q":
        return Angle1QFunction(L=L)
    if family_norm == "fingerprint":
        if n is None:
            raise ValueError("Fingerprint family requires --n.")
        return FingerprintPhaseFunction(L=L, n=n)
    raise ValueError(f"Unsupported QOWF family '{family}'. Use angle1q or fingerprint.")


__all__ = [
    "QuantumOneWayFunction",
    "Angle1QFunction",
    "FingerprintPhaseFunction",
    "build_qowf",
]
