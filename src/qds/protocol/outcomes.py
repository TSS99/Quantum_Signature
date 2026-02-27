from __future__ import annotations

from enum import Enum


class VerificationOutcome(str, Enum):
    ONE_ACC = "ONE_ACC"
    ZERO_ACC = "ZERO_ACC"
    REJ = "REJ"
