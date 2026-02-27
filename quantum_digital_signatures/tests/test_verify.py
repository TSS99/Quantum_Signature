from __future__ import annotations

import pytest

from qds.qowf import build_qowf
from qds.quantum.verify import verify_by_circuit, verify_statevector


def test_verify_statevector_accepts_matching_key() -> None:
    qowf = build_qowf(family="angle1q", L=6, n=1)
    key = "001011"
    psi = qowf.statevector(key)
    prob = verify_statevector(psi, key, qowf)
    assert prob == pytest.approx(1.0, abs=1e-12)


def test_verify_statevector_rejects_wrong_key_with_nonzero_gap() -> None:
    qowf = build_qowf(family="angle1q", L=6, n=1)
    psi = qowf.statevector("000000")
    wrong_key = "111111"
    prob = verify_statevector(psi, wrong_key, qowf)
    assert prob < 1.0


def test_verify_by_circuit_exact_matches_matching_key() -> None:
    qowf = build_qowf(family="fingerprint", L=6, n=3)
    key = "101010"
    prep = qowf.prepare_circuit(key)
    prob = verify_by_circuit(prep, key, qowf, shots=None)
    assert prob == pytest.approx(1.0, abs=1e-12)
