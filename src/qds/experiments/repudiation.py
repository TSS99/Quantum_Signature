from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qds.protocol.distribution import CopyLedger, DistributionPayload, direct_distribute, distributed_swap_test_two_recipients
from qds.protocol.entities import Recipient, SignedMessage
from qds.protocol.outcomes import VerificationOutcome
from qds.qowf import build_qowf
from qds.quantum.utils import ensure_artifacts_dir, seeded_rng


def _random_key(L: int, rng: np.random.Generator) -> str:
    return format(int(rng.integers(0, 2**L)), f"0{L}b")


def _distinct_key(*, L: int, base_key: str, rng: np.random.Generator) -> str:
    candidate = _random_key(L, rng)
    while candidate == base_key:
        candidate = _random_key(L, rng)
    return candidate


def run_repudiation_experiment(
    *,
    trials: int = 200,
    M: int = 64,
    c1: float = 0.05,
    c2: float = 0.20,
    T_max: int = 4,
    family: str = "angle1q",
    L: int = 8,
    n: int = 1,
    with_distributed_swap_test: bool = False,
    verification_mode: str = "uncompute",
    shots: int | None = 1,
    seed: int | None = 19,
    artifacts_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    if M <= 0:
        raise ValueError("M must be positive.")
    if with_distributed_swap_test and T_max < 4:
        raise ValueError("Distributed swap-test setup requires T_max >= 4.")

    rng = seeded_rng(seed)
    rows: list[dict[str, Any]] = []

    for trial in range(trials):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        trial_rng = np.random.default_rng(trial_seed)
        qowf = build_qowf(family=family, L=L, n=n)
        bob = Recipient(name="bob", qowf=qowf, M=M, rng=np.random.default_rng(trial_seed + 1))
        charlie = Recipient(name="charlie", qowf=qowf, M=M, rng=np.random.default_rng(trial_seed + 2))

        signed_bit = int(trial_rng.integers(0, 2))
        other_bit = 1 - signed_bit
        revealed_keys: list[str] = []
        payload: DistributionPayload = {}
        copies = 2 if with_distributed_swap_test else 1

        for i in range(M):
            good_key = _random_key(L, trial_rng)
            bad_key = _distinct_key(L=L, base_key=good_key, rng=trial_rng)
            common_other = _random_key(L, trial_rng)
            revealed_keys.append(good_key)

            payload[(signed_bit, i)] = {
                "bob": [qowf.statevector(good_key) for _ in range(copies)],
                "charlie": [qowf.statevector(bad_key) for _ in range(copies)],
            }
            payload[(other_bit, i)] = {
                "bob": [qowf.statevector(common_other) for _ in range(copies)],
                "charlie": [qowf.statevector(common_other) for _ in range(copies)],
            }

        ledger = CopyLedger(T_max=T_max)
        distribution_success = True
        if with_distributed_swap_test:
            distribution_success = distributed_swap_test_two_recipients(
                payload=payload,
                bob=bob,
                charlie=charlie,
                ledger=ledger,
                rng=trial_rng,
                sampled=False,
                shots=1,
            )
        else:
            direct_distribute(payload=payload, recipients={"bob": bob, "charlie": charlie}, ledger=ledger)

        if not distribution_success:
            rows.append(
                {
                    "trial": trial,
                    "seed": trial_seed,
                    "distribution_success": 0,
                    "bob_outcome": "ABORT",
                    "charlie_outcome": "ABORT",
                    "disagreement": 0,
                    "targeted_repudiation_success": 0,
                }
            )
            continue

        signed_message = SignedMessage(bit=signed_bit, revealed_keys=revealed_keys)
        bob_result = bob.verify_signature(
            signed_message=signed_message,
            c1=c1,
            c2=c2,
            verification_mode=verification_mode,
            shots=shots,
        )
        charlie_result = charlie.verify_signature(
            signed_message=signed_message,
            c1=c1,
            c2=c2,
            verification_mode=verification_mode,
            shots=shots,
        )
        disagreement = int(bob_result.outcome != charlie_result.outcome)
        targeted = int(
            bob_result.outcome in {VerificationOutcome.ONE_ACC, VerificationOutcome.ZERO_ACC}
            and charlie_result.outcome == VerificationOutcome.REJ
        )
        rows.append(
            {
                "trial": trial,
                "seed": trial_seed,
                "distribution_success": 1,
                "bob_outcome": bob_result.outcome.value,
                "charlie_outcome": charlie_result.outcome.value,
                "disagreement": disagreement,
                "targeted_repudiation_success": targeted,
            }
        )

    df = pd.DataFrame(rows)
    out_dir = ensure_artifacts_dir(artifacts_dir)
    suffix = "with_dist_swap" if with_distributed_swap_test else "no_dist_swap"
    csv_path = out_dir / f"repudiation_{suffix}.csv"
    plot_path = out_dir / f"repudiation_{suffix}.png"
    df.to_csv(csv_path, index=False)

    abort_rate = float((df["distribution_success"] == 0).mean())
    non_abort = df[df["distribution_success"] == 1]
    disagreement_non_abort = float(non_abort["disagreement"].mean()) if len(non_abort) else 0.0
    targeted_non_abort = float(non_abort["targeted_repudiation_success"].mean()) if len(non_abort) else 0.0

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = ["Abort", "Agree", "Disagree"]
    if len(non_abort):
        agree_rate = 1.0 - disagreement_non_abort
    else:
        agree_rate = 0.0
    values = [abort_rate, (1 - abort_rate) * agree_rate, (1 - abort_rate) * disagreement_non_abort]
    bars = ax.bar(labels, values, color=["#7f8c8d", "#27ae60", "#e74c3c"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title(
        "Repudiation/Transferability Outcomes "
        + ("with Distributed Swap-Test" if with_distributed_swap_test else "without Distributed Swap-Test")
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    summary = {
        "trials": trials,
        "with_distributed_swap_test": with_distributed_swap_test,
        "abort_rate": abort_rate,
        "disagreement_rate_non_abort": disagreement_non_abort,
        "targeted_repudiation_rate_non_abort": targeted_non_abort,
    }
    return {"summary": summary, "csv_path": str(csv_path), "plot_path": str(plot_path)}
