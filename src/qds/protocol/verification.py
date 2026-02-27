from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

from qds.protocol.outcomes import VerificationOutcome


@dataclass(frozen=True)
class RecipientVerificationResult:
    recipient: str
    failures: int
    threshold_one: int
    threshold_reject: int
    outcome: VerificationOutcome
    per_key_accepts: list[bool]


@dataclass(frozen=True)
class DecisionThresholdResult:
    failures: int
    threshold_one: int
    threshold_reject: int
    outcome: VerificationOutcome


def decide_outcome(*, failures: int, M: int, c1: float, c2: float) -> DecisionThresholdResult:
    if M <= 0:
        raise ValueError("M must be positive.")
    if not (0 <= c1 < c2 <= 1):
        raise ValueError("Thresholds must satisfy 0 <= c1 < c2 <= 1.")
    if failures < 0 or failures > M:
        raise ValueError("failures must be in [0, M].")

    threshold_one = floor(c1 * M)
    threshold_reject = ceil(c2 * M)
    if failures <= threshold_one:
        outcome = VerificationOutcome.ONE_ACC
    elif failures >= threshold_reject:
        outcome = VerificationOutcome.REJ
    else:
        outcome = VerificationOutcome.ZERO_ACC

    return DecisionThresholdResult(
        failures=failures,
        threshold_one=threshold_one,
        threshold_reject=threshold_reject,
        outcome=outcome,
    )
