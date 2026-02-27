from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from qds.protocol.distribution import (
    CopyLedger,
    DistributionPayload,
    direct_distribute,
    distributed_swap_test_two_recipients,
    make_honest_payload,
    trusted_distributor_forward,
)
from qds.protocol.entities import Alice, Recipient, SignedMessage
from qds.protocol.outcomes import VerificationOutcome
from qds.qowf import build_qowf
from qds.quantum.utils import seeded_rng


@dataclass(frozen=True)
class ProtocolConfig:
    M: int = 32
    c1: float = 0.05
    c2: float = 0.20
    T_max: int = 4
    L: int = 8
    n: int = 1
    family: str = "angle1q"
    verification_mode: str = "uncompute"
    shots: int | None = 1
    seed: int | None = None

    def validate(self) -> None:
        if self.M <= 0:
            raise ValueError("M must be positive.")
        if not (0 <= self.c1 < self.c2 <= 1):
            raise ValueError("Thresholds must satisfy 0 <= c1 < c2 <= 1.")
        if self.T_max <= 0:
            raise ValueError("T_max must be positive.")
        if self.verification_mode not in {"uncompute", "swap-test"}:
            raise ValueError("verification_mode must be either 'uncompute' or 'swap-test'.")
        if self.shots is not None and self.shots <= 0:
            raise ValueError("shots must be positive when provided.")
        if self.family == "angle1q" and self.n != 1:
            raise ValueError("angle1q uses n=1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _new_round_entities(
    *,
    config: ProtocolConfig,
    recipient_names: list[str],
) -> tuple[Alice, dict[str, Recipient], CopyLedger, np.random.Generator]:
    config.validate()
    rng = seeded_rng(config.seed)
    qowf = build_qowf(family=config.family, L=config.L, n=config.n)
    alice = Alice(qowf=qowf, M=config.M, rng=rng)
    alice.generate_keys()

    recipients: dict[str, Recipient] = {}
    for idx, name in enumerate(recipient_names):
        child_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + idx)
        recipients[name] = Recipient(name=name, qowf=qowf, M=config.M, rng=child_rng)
    ledger = CopyLedger(T_max=config.T_max)
    return alice, recipients, ledger, rng


def generate_keygen_bundle(
    *,
    config: ProtocolConfig,
    recipient_names: list[str],
) -> dict[str, Any]:
    alice, _recipients, _ledger, _rng = _new_round_entities(config=config, recipient_names=recipient_names)
    _payload, recipient_labels = make_honest_payload(
        alice=alice,
        recipient_names=recipient_names,
        copies_per_recipient=1,
    )
    return {
        "params": config.to_dict(),
        "alice_private_keys": {
            "0": list(alice.private_keys[0]),
            "1": list(alice.private_keys[1]),
        },
        "recipient_public_keys": recipient_labels,
    }


def run_single_bit_round(
    *,
    bit: int,
    config: ProtocolConfig,
    recipient_names: list[str],
    distribution_mode: str = "direct",
    distribution_sampled: bool = False,
    distribution_shots: int = 1,
    noise_model=None,
    payload_override: DistributionPayload | None = None,
    recipient_labels_override: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, Any]:
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1.")
    if not recipient_names:
        raise ValueError("recipient_names cannot be empty.")

    alice, recipients, ledger, rng = _new_round_entities(config=config, recipient_names=recipient_names)
    if payload_override is None:
        copies = 2 if distribution_mode == "distributed-swap-test" else 1
        payload, recipient_labels = make_honest_payload(
            alice=alice,
            recipient_names=recipient_names,
            copies_per_recipient=copies,
        )
    else:
        payload = payload_override
        recipient_labels = recipient_labels_override or {}

    distribution_success = True
    if distribution_mode == "direct":
        direct_distribute(payload=payload, recipients=recipients, ledger=ledger)
    elif distribution_mode == "trusted":
        trusted_distributor_forward(
            payload=payload,
            recipients=recipients,
            ledger=ledger,
            rng=rng,
            sampled=distribution_sampled,
            shots=distribution_shots,
            noise_model=noise_model,
        )
    elif distribution_mode == "distributed-swap-test":
        if len(recipient_names) != 2:
            raise ValueError("distributed-swap-test mode currently supports exactly 2 recipients.")
        bob = recipients[recipient_names[0]]
        charlie = recipients[recipient_names[1]]
        distribution_success = distributed_swap_test_two_recipients(
            payload=payload,
            bob=bob,
            charlie=charlie,
            ledger=ledger,
            rng=rng,
            sampled=distribution_sampled,
            shots=distribution_shots,
            noise_model=noise_model,
        )
    else:
        raise ValueError("distribution_mode must be one of: direct, trusted, distributed-swap-test.")

    signed_message = alice.sign(bit)
    recipient_results = []
    if distribution_success:
        for name in recipient_names:
            result = recipients[name].verify_signature(
                signed_message=signed_message,
                c1=config.c1,
                c2=config.c2,
                verification_mode=config.verification_mode,
                shots=config.shots,
                noise_model=noise_model,
            )
            recipient_results.append(
                {
                    "recipient": result.recipient,
                    "s_j": result.failures,
                    "outcome": result.outcome.value,
                    "threshold_one": result.threshold_one,
                    "threshold_reject": result.threshold_reject,
                }
            )

    return {
        "params": config.to_dict(),
        "distribution_mode": distribution_mode,
        "distribution_success": distribution_success,
        "signed_message": signed_message.to_dict(),
        "recipient_results": recipient_results,
        "recipient_public_keys": recipient_labels,
        "alice_private_keys": {"0": list(alice.private_keys[0]), "1": list(alice.private_keys[1])},
    }


def encode_repetition(message_bits: str, repetition: int) -> str:
    if repetition <= 0:
        raise ValueError("repetition must be positive.")
    if not message_bits or any(ch not in {"0", "1"} for ch in message_bits):
        raise ValueError("message_bits must be a non-empty binary string.")
    return "".join(ch * repetition for ch in message_bits)


def run_multibit_repetition(
    *,
    message_bits: str,
    repetition: int,
    config: ProtocolConfig,
    recipient_names: list[str],
    encoded_mode: bool = False,
) -> dict[str, Any]:
    encoded_bits = encode_repetition(message_bits, repetition) if encoded_mode else message_bits
    rounds = []
    for round_index, bit_char in enumerate(encoded_bits):
        local_seed = None if config.seed is None else config.seed + round_index
        local_cfg = ProtocolConfig(
            M=config.M,
            c1=config.c1,
            c2=config.c2,
            T_max=config.T_max,
            L=config.L,
            n=config.n,
            family=config.family,
            verification_mode=config.verification_mode,
            shots=config.shots,
            seed=local_seed,
        )
        rounds.append(
            run_single_bit_round(
                bit=int(bit_char),
                config=local_cfg,
                recipient_names=recipient_names,
                distribution_mode="direct",
            )
        )
    return {
        "message_bits": message_bits,
        "encoded_mode": encoded_mode,
        "repetition": repetition,
        "encoded_bits": encoded_bits,
        "rounds": rounds,
    }


def recipient_outcome_counts(round_result: dict[str, Any]) -> dict[str, int]:
    counts = {VerificationOutcome.ONE_ACC.value: 0, VerificationOutcome.ZERO_ACC.value: 0, VerificationOutcome.REJ.value: 0}
    for item in round_result.get("recipient_results", []):
        outcome = item["outcome"]
        if outcome in counts:
            counts[outcome] += 1
    return counts
