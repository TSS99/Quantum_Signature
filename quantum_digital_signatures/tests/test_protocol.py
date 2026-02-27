from __future__ import annotations

import numpy as np
import pytest

from qds.protocol.distribution import CopyLedger
from qds.protocol.entities import Alice, CopyLimitError, Recipient, SingleUseError
from qds.protocol.outcomes import VerificationOutcome
from qds.protocol.verification import decide_outcome
from qds.qowf import build_qowf


def test_decision_thresholds_cover_three_outcomes() -> None:
    M, c1, c2 = 10, 0.1, 0.3
    one = decide_outcome(failures=1, M=M, c1=c1, c2=c2)
    zero = decide_outcome(failures=2, M=M, c1=c1, c2=c2)
    rej = decide_outcome(failures=3, M=M, c1=c1, c2=c2)
    assert one.outcome == VerificationOutcome.ONE_ACC
    assert zero.outcome == VerificationOutcome.ZERO_ACC
    assert rej.outcome == VerificationOutcome.REJ


def test_copy_ledger_enforces_t_max() -> None:
    ledger = CopyLedger(T_max=2)
    key_id = (0, 0)
    ledger.issue(key_id, copies=2)
    with pytest.raises(CopyLimitError):
        ledger.issue(key_id, copies=1)


def test_single_use_enforced_after_verification() -> None:
    qowf = build_qowf(family="angle1q", L=6, n=1)
    rng = np.random.default_rng(5)
    alice = Alice(qowf=qowf, M=4, rng=rng)
    alice.generate_keys()
    recipient = Recipient(name="bob", qowf=qowf, M=4, rng=np.random.default_rng(6))

    for bit in (0, 1):
        for i, key in enumerate(alice.private_keys[bit]):
            recipient.add_public_key_copy(bit=bit, index=i, state=qowf.statevector(key))

    signed = alice.sign(0)
    result = recipient.verify_signature(
        signed_message=signed,
        c1=0.0,
        c2=0.5,
        verification_mode="uncompute",
        shots=1,
    )
    assert result.outcome in {
        VerificationOutcome.ONE_ACC,
        VerificationOutcome.ZERO_ACC,
        VerificationOutcome.REJ,
    }

    with pytest.raises(SingleUseError):
        recipient.verify_signature(
            signed_message=signed,
            c1=0.0,
            c2=0.5,
            verification_mode="uncompute",
            shots=1,
        )
