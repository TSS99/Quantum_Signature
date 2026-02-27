from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector

from qds.protocol.entities import Alice, Recipient, SignedMessage
from qds.protocol.outcomes import VerificationOutcome
from qds.qowf import build_qowf
from qds.quantum.utils import ensure_artifacts_dir, seeded_rng


def _sample_one_qubit_basis_measurement(
    state: Statevector, *, basis: str, rng: np.random.Generator
) -> int:
    if state.num_qubits != 1:
        raise ValueError("Angle-family estimator expects one-qubit states.")
    amp0, amp1 = state.data
    if basis == "Z":
        p1 = float(abs(amp1) ** 2)
    elif basis == "X":
        plus = (amp0 + amp1) / np.sqrt(2.0)
        minus = (amp0 - amp1) / np.sqrt(2.0)
        p1 = float(abs(minus) ** 2)
    else:
        raise ValueError("basis must be 'Z' or 'X'.")
    return int(rng.random() < p1)


def estimate_angle_key_from_measurements(
    *,
    state: Statevector,
    T: int,
    L: int,
    rng: np.random.Generator,
) -> str:
    if T <= 0:
        raise ValueError("T must be positive.")

    z_obs: list[int] = []
    x_obs: list[int] = []
    for _ in range(T):
        basis = "Z" if rng.random() < 0.5 else "X"
        bit = _sample_one_qubit_basis_measurement(state, basis=basis, rng=rng)
        value = 1 - 2 * bit  # map outcome 0->+1, 1->-1
        if basis == "Z":
            z_obs.append(value)
        else:
            x_obs.append(value)

    e_z = float(np.mean(z_obs)) if z_obs else 0.0
    e_x = float(np.mean(x_obs)) if x_obs else 0.0
    theta = np.pi / (2**L)
    angle_hat = 0.5 * np.arctan2(e_x, e_z)
    if angle_hat < 0:
        angle_hat += np.pi
    j_hat = int(np.clip(np.rint(angle_hat / theta), 0, 2**L - 1))
    return format(j_hat, f"0{L}b")


def guess_fingerprint_key_naive(
    *,
    state: Statevector,
    T: int,
    L: int,
    rng: np.random.Generator,
) -> str:
    if T <= 0:
        raise ValueError("T must be positive.")
    probs = np.abs(state.data) ** 2
    samples = [str(int(rng.choice(len(probs), p=probs))) for _ in range(T)]
    digest = hashlib.sha256(",".join(samples).encode("ascii")).digest()
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
    if bits.size < L:
        reps = int(np.ceil(L / bits.size))
        bits = np.tile(bits, reps)
    return "".join(str(int(b)) for b in bits[:L])


def run_forgery_experiment(
    *,
    trials: int = 200,
    M: int = 64,
    T: int = 4,
    c1: float = 0.05,
    c2: float = 0.20,
    T_max: int = 5,
    family: str = "angle1q",
    L: int = 8,
    n: int = 1,
    verification_mode: str = "uncompute",
    shots: int | None = 1,
    seed: int | None = 7,
    artifacts_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    if M <= 0:
        raise ValueError("M must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if T + 1 > T_max:
        raise ValueError(
            f"Forging setup requires T+1 copies per key for target bit (T={T}); choose T_max >= {T + 1}."
        )

    rng = seeded_rng(seed)
    rows: list[dict[str, Any]] = []

    for trial in range(trials):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        trial_rng = np.random.default_rng(trial_seed)
        qowf = build_qowf(family=family, L=L, n=n)
        alice = Alice(qowf=qowf, M=M, rng=trial_rng)
        alice.generate_keys()
        bob = Recipient(name="bob", qowf=qowf, M=M, rng=np.random.default_rng(trial_seed + 1))

        signed_bit = int(trial_rng.integers(0, 2))
        target_bit = 1 - signed_bit
        guessed_keys: list[str] = []
        for i in range(M):
            true_key = alice.private_keys[target_bit][i]
            target_state = qowf.statevector(true_key)
            bob.add_public_key_copy(bit=target_bit, index=i, state=target_state)
            if family == "angle1q":
                guess = estimate_angle_key_from_measurements(
                    state=target_state,
                    T=T,
                    L=L,
                    rng=trial_rng,
                )
            else:
                guess = guess_fingerprint_key_naive(
                    state=target_state,
                    T=T,
                    L=L,
                    rng=trial_rng,
                )
            guessed_keys.append(guess)

        forged_signature = SignedMessage(bit=target_bit, revealed_keys=guessed_keys)
        result = bob.verify_signature(
            signed_message=forged_signature,
            c1=c1,
            c2=c2,
            verification_mode=verification_mode,
            shots=shots,
        )
        rows.append(
            {
                "trial": trial,
                "seed": trial_seed,
                "signed_bit": signed_bit,
                "forged_bit": target_bit,
                "s_j": result.failures,
                "outcome": result.outcome.value,
                "any_accept": int(result.outcome != VerificationOutcome.REJ),
                "transferable_success": int(result.outcome == VerificationOutcome.ONE_ACC),
            }
        )

    df = pd.DataFrame(rows)
    out_dir = ensure_artifacts_dir(artifacts_dir)
    csv_path = out_dir / "forgery_results.csv"
    plot_path = out_dir / "forgery_results.png"
    df.to_csv(csv_path, index=False)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = [
        VerificationOutcome.ONE_ACC.value,
        VerificationOutcome.ZERO_ACC.value,
        VerificationOutcome.REJ.value,
    ]
    rates = df["outcome"].value_counts(normalize=True).reindex(order, fill_value=0.0)
    bars = ax.bar(order, rates.values, color=["#2e86de", "#f39c12", "#c0392b"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title(f"Forgery Outcomes ({family}, M={M}, T={T}, trials={trials})")
    for bar, value in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    summary = {
        "trials": trials,
        "forgery_any_accept_rate": float(df["any_accept"].mean()),
        "forgery_transferable_rate": float(df["transferable_success"].mean()),
        "mean_failures": float(df["s_j"].mean()),
    }
    return {"summary": summary, "csv_path": str(csv_path), "plot_path": str(plot_path)}
