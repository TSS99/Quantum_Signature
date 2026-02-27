from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
from qiskit.quantum_info import Statevector

from qds.protocol.outcomes import VerificationOutcome
from qds.protocol.verification import RecipientVerificationResult, decide_outcome
from qds.qowf.base import QuantumOneWayFunction
from qds.quantum.swap_test import swap_test_pass_probability_exact, swap_test_pass_probability_sampled
from qds.quantum.utils import statevector_to_circuit
from qds.quantum.verify import verify_by_circuit, verify_statevector

KeyId = Tuple[int, int]


class CopyLimitError(RuntimeError):
    """Raised when simulator copy limit T_max is exceeded."""


class SingleUseError(RuntimeError):
    """Raised when trying to re-use consumed single-use key material."""


@dataclass(frozen=True)
class SignedMessage:
    bit: int
    revealed_keys: list[str]

    def to_dict(self) -> dict:
        return {"bit": self.bit, "revealed_keys": list(self.revealed_keys)}

    @staticmethod
    def from_dict(data: dict) -> "SignedMessage":
        return SignedMessage(bit=int(data["bit"]), revealed_keys=[str(x) for x in data["revealed_keys"]])


class Alice:
    """Alice key generation and signing entity."""

    def __init__(self, *, qowf: QuantumOneWayFunction, M: int, rng: np.random.Generator) -> None:
        if M <= 0:
            raise ValueError("M must be positive.")
        self.qowf = qowf
        self.M = M
        self.rng = rng
        self.private_keys: Dict[int, List[str]] = {0: [], 1: []}

    def generate_keys(self) -> None:
        self.private_keys = {0: [], 1: []}
        for b in (0, 1):
            for _ in range(self.M):
                value = int(self.rng.integers(0, 2**self.qowf.L))
                self.private_keys[b].append(format(value, f"0{self.qowf.L}b"))

    def sign(self, bit: int) -> SignedMessage:
        if bit not in (0, 1):
            raise ValueError("Bit message must be 0 or 1.")
        keys = self.private_keys.get(bit, [])
        if len(keys) != self.M:
            raise RuntimeError("Keys not generated. Call generate_keys() before sign().")
        return SignedMessage(bit=bit, revealed_keys=list(keys))


class Recipient:
    """Recipient holding single-use public key copies for one message round."""

    def __init__(self, *, name: str, qowf: QuantumOneWayFunction, M: int, rng: np.random.Generator) -> None:
        self.name = name
        self.qowf = qowf
        self.M = M
        self.rng = rng
        self.public_key_copies: DefaultDict[KeyId, List[Statevector]] = defaultdict(list)
        self.round_consumed = False

    def add_public_key_copy(self, *, bit: int, index: int, state: Statevector) -> None:
        if self.round_consumed:
            raise SingleUseError(f"Recipient {self.name} already consumed this round's keys.")
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1.")
        if index < 0 or index >= self.M:
            raise ValueError(f"index must be in [0, {self.M - 1}].")
        if state.dim != self.qowf.statevector(0).dim:
            raise ValueError("State dimension does not match qowf dimension.")
        self.public_key_copies[(bit, index)].append(state.copy())

    def available_copies(self, *, bit: int, index: int) -> int:
        return len(self.public_key_copies[(bit, index)])

    def load_public_key_labels(self, labels_by_bit: dict[str, list[str]]) -> None:
        for bit_str, labels in labels_by_bit.items():
            bit = int(bit_str)
            if len(labels) != self.M:
                raise ValueError(
                    f"Recipient {self.name} expected M={self.M} labels for bit={bit}, got {len(labels)}."
                )
            for index, key_label in enumerate(labels):
                self.add_public_key_copy(bit=bit, index=index, state=self.qowf.statevector(key_label))

    def _consume_public_key_copy(self, *, bit: int, index: int) -> Statevector:
        copies = self.public_key_copies[(bit, index)]
        if not copies:
            raise SingleUseError(
                f"Recipient {self.name} has no remaining public-key copies for bit={bit}, index={index}."
            )
        return copies.pop(0)

    def discard_round_keys(self) -> None:
        self.public_key_copies.clear()
        self.round_consumed = True

    def verify_signature(
        self,
        *,
        signed_message: SignedMessage,
        c1: float,
        c2: float,
        verification_mode: str = "uncompute",
        shots: int | None = 1,
        noise_model=None,
    ) -> RecipientVerificationResult:
        if self.round_consumed:
            raise SingleUseError(f"Recipient {self.name} has already verified with these single-use keys.")
        if len(signed_message.revealed_keys) != self.M:
            raise ValueError(
                f"Signed message carries {len(signed_message.revealed_keys)} keys, expected M={self.M}."
            )
        if verification_mode not in {"uncompute", "swap-test"}:
            raise ValueError("verification_mode must be 'uncompute' or 'swap-test'.")

        failures = 0
        per_key_accepts: list[bool] = []
        for index, claimed_key in enumerate(signed_message.revealed_keys):
            public_state = self._consume_public_key_copy(bit=signed_message.bit, index=index)
            accepted = self._verify_single_key(
                public_state=public_state,
                claimed_key=claimed_key,
                verification_mode=verification_mode,
                shots=shots,
                noise_model=noise_model,
            )
            per_key_accepts.append(accepted)
            if not accepted:
                failures += 1

        decision = decide_outcome(failures=failures, M=self.M, c1=c1, c2=c2)
        self.discard_round_keys()
        return RecipientVerificationResult(
            recipient=self.name,
            failures=decision.failures,
            threshold_one=decision.threshold_one,
            threshold_reject=decision.threshold_reject,
            outcome=decision.outcome,
            per_key_accepts=per_key_accepts,
        )

    def _verify_single_key(
        self,
        *,
        public_state: Statevector,
        claimed_key: str,
        verification_mode: str,
        shots: int | None,
        noise_model,
    ) -> bool:
        seed_local = int(self.rng.integers(0, 2**31 - 1))
        if verification_mode == "uncompute":
            if shots is None:
                accept_prob = verify_statevector(public_state, claimed_key, self.qowf)
                return bool(self.rng.random() < accept_prob)
            prep = statevector_to_circuit(public_state)
            accept_prob = verify_by_circuit(
                prep,
                claimed_key,
                self.qowf,
                shots=max(1, shots),
                seed=seed_local,
                noise_model=noise_model,
            )
            if shots == 1:
                return accept_prob >= 1.0
            return bool(self.rng.random() < accept_prob)

        target = self.qowf.statevector(claimed_key)
        if shots is None:
            accept_prob = swap_test_pass_probability_exact(public_state, target)
            return bool(self.rng.random() < accept_prob)
        accept_prob, _ = swap_test_pass_probability_sampled(
            public_state, target, shots=max(1, shots), seed=seed_local, noise_model=noise_model
        )
        if shots == 1:
            return accept_prob >= 1.0
        return bool(self.rng.random() < accept_prob)
