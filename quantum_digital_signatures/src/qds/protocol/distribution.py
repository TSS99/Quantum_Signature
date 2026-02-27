from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from qiskit.quantum_info import Statevector

from qds.protocol.entities import Alice, CopyLimitError, Recipient
from qds.quantum.swap_test import swap_test_pass_probability_exact, swap_test_pass_probability_sampled

KeyId = Tuple[int, int]
DistributionPayload = Dict[KeyId, Dict[str, List[Statevector]]]


@dataclass
class CopyLedger:
    T_max: int
    issued: dict[KeyId, int] = field(default_factory=lambda: defaultdict(int))

    def issue(self, key_id: KeyId, copies: int = 1) -> None:
        if copies <= 0:
            raise ValueError("copies must be positive.")
        current = self.issued[key_id]
        if current + copies > self.T_max:
            raise CopyLimitError(
                f"Copy limit exceeded for key={key_id}: requested {current + copies}, T_max={self.T_max}."
            )
        self.issued[key_id] = current + copies


def make_honest_payload(
    *,
    alice: Alice,
    recipient_names: list[str],
    copies_per_recipient: int = 1,
) -> tuple[DistributionPayload, dict[str, dict[str, list[str]]]]:
    if copies_per_recipient <= 0:
        raise ValueError("copies_per_recipient must be positive.")

    payload: DistributionPayload = {}
    labels: dict[str, dict[str, list[str]]] = {
        name: {"0": [], "1": []} for name in recipient_names
    }
    for bit in (0, 1):
        for index, key in enumerate(alice.private_keys[bit]):
            state = alice.qowf.statevector(key)
            payload[(bit, index)] = {}
            for name in recipient_names:
                payload[(bit, index)][name] = [state.copy() for _ in range(copies_per_recipient)]
                labels[name][str(bit)].append(key)
    return payload, labels


def direct_distribute(
    *,
    payload: DistributionPayload,
    recipients: dict[str, Recipient],
    ledger: CopyLedger,
) -> None:
    for key_id, recipient_map in payload.items():
        for recipient_name, copies in recipient_map.items():
            recipient = recipients.get(recipient_name)
            if recipient is None:
                raise ValueError(f"Unknown recipient in payload: {recipient_name}.")
            for state in copies:
                ledger.issue(key_id, copies=1)
                recipient.add_public_key_copy(bit=key_id[0], index=key_id[1], state=state)


def _swap_test_one_shot(
    *,
    state_a: Statevector,
    state_b: Statevector,
    rng: np.random.Generator,
    sampled: bool,
    shots: int,
    noise_model=None,
) -> bool:
    if sampled:
        pass_prob, _ = swap_test_pass_probability_sampled(
            state_a,
            state_b,
            shots=max(1, shots),
            seed=int(rng.integers(0, 2**31 - 1)),
            noise_model=noise_model,
        )
        if shots == 1:
            return pass_prob >= 1.0
        return bool(rng.random() < pass_prob)

    pass_prob = swap_test_pass_probability_exact(state_a, state_b)
    return bool(rng.random() < pass_prob)


def trusted_distributor_forward(
    *,
    payload: DistributionPayload,
    recipients: dict[str, Recipient],
    ledger: CopyLedger,
    rng: np.random.Generator,
    sampled: bool = False,
    shots: int = 1,
    noise_model=None,
) -> None:
    for key_id, recipient_map in payload.items():
        flat_copies: list[Statevector] = [state for copies in recipient_map.values() for state in copies]
        if len(flat_copies) <= 1:
            continue
        reference = flat_copies[0]
        for candidate in flat_copies[1:]:
            passed = _swap_test_one_shot(
                state_a=reference,
                state_b=candidate,
                rng=rng,
                sampled=sampled,
                shots=shots,
                noise_model=noise_model,
            )
            if not passed:
                raise RuntimeError(f"Trusted distributor aborted: swap-test failed for key {key_id}.")
    direct_distribute(payload=payload, recipients=recipients, ledger=ledger)


def distributed_swap_test_two_recipients(
    *,
    payload: DistributionPayload,
    bob: Recipient,
    charlie: Recipient,
    ledger: CopyLedger,
    rng: np.random.Generator,
    sampled: bool = False,
    shots: int = 1,
    noise_model=None,
) -> bool:
    kept_for_bob: dict[KeyId, Statevector] = {}
    kept_for_charlie: dict[KeyId, Statevector] = {}

    for key_id, recipient_map in payload.items():
        bob_copies = recipient_map.get(bob.name, [])
        charlie_copies = recipient_map.get(charlie.name, [])
        if len(bob_copies) < 2 or len(charlie_copies) < 2:
            raise ValueError(
                f"Distributed swap-test requires 2 copies for each recipient at key={key_id}."
            )

        total_issued = len(bob_copies) + len(charlie_copies)
        ledger.issue(key_id, copies=total_issued)

        local_bob = _swap_test_one_shot(
            state_a=bob_copies[0],
            state_b=bob_copies[1],
            rng=rng,
            sampled=sampled,
            shots=shots,
            noise_model=noise_model,
        )
        if not local_bob:
            return False

        local_charlie = _swap_test_one_shot(
            state_a=charlie_copies[0],
            state_b=charlie_copies[1],
            rng=rng,
            sampled=sampled,
            shots=shots,
            noise_model=noise_model,
        )
        if not local_charlie:
            return False

        cross_check = _swap_test_one_shot(
            state_a=bob_copies[1],
            state_b=charlie_copies[1],
            rng=rng,
            sampled=sampled,
            shots=shots,
            noise_model=noise_model,
        )
        if not cross_check:
            return False

        kept_for_bob[key_id] = bob_copies[0]
        kept_for_charlie[key_id] = charlie_copies[0]

    for (bit, idx), state in kept_for_bob.items():
        bob.add_public_key_copy(bit=bit, index=idx, state=state)
    for (bit, idx), state in kept_for_charlie.items():
        charlie.add_public_key_copy(bit=bit, index=idx, state=state)
    return True
