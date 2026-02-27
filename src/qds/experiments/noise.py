from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qds.protocol.outcomes import VerificationOutcome
from qds.protocol.signing import ProtocolConfig, run_single_bit_round
from qds.quantum.utils import build_noise_model, ensure_artifacts_dir, seeded_rng


def run_noise_experiment(
    *,
    trials: int = 200,
    M: int = 64,
    c1_values: list[float] | tuple[float, ...] = (0.0, 0.05, 0.1),
    c2: float = 0.20,
    T_max: int = 4,
    family: str = "angle1q",
    L: int = 8,
    n: int = 1,
    depolarizing_prob: float = 0.01,
    readout_error_prob: float = 0.02,
    seed: int | None = 11,
    artifacts_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    c1_values = list(c1_values)
    if not c1_values:
        raise ValueError("c1_values must be non-empty.")

    noise_model = build_noise_model(
        depolarizing_prob=depolarizing_prob,
        readout_error_prob=readout_error_prob,
    )
    rng = seeded_rng(seed)
    rows: list[dict[str, Any]] = []

    for c1 in c1_values:
        if not (0 <= c1 < c2):
            raise ValueError(f"Each c1 must satisfy 0 <= c1 < c2. Bad value: c1={c1}, c2={c2}.")
        for trial in range(trials):
            trial_seed = int(rng.integers(0, 2**31 - 1))
            bit = int(rng.integers(0, 2))
            config = ProtocolConfig(
                M=M,
                c1=c1,
                c2=c2,
                T_max=T_max,
                L=L,
                n=n,
                family=family,
                verification_mode="uncompute",
                shots=1,
                seed=trial_seed,
            )
            out = run_single_bit_round(
                bit=bit,
                config=config,
                recipient_names=["bob"],
                distribution_mode="direct",
                noise_model=noise_model,
            )
            bob_result = out["recipient_results"][0]
            rows.append(
                {
                    "trial": trial,
                    "seed": trial_seed,
                    "c1": c1,
                    "c2": c2,
                    "bit": bit,
                    "s_j": bob_result["s_j"],
                    "outcome": bob_result["outcome"],
                    "reject": int(bob_result["outcome"] == VerificationOutcome.REJ.value),
                    "one_acc": int(bob_result["outcome"] == VerificationOutcome.ONE_ACC.value),
                }
            )

    df = pd.DataFrame(rows)
    out_dir = ensure_artifacts_dir(artifacts_dir)
    csv_path = out_dir / "noise_experiment.csv"
    plot_path = out_dir / "noise_experiment.png"
    df.to_csv(csv_path, index=False)

    grouped = (
        df.groupby("c1")
        .agg(
            mean_sj=("s_j", "mean"),
            reject_rate=("reject", "mean"),
            one_acc_rate=("one_acc", "mean"),
        )
        .reset_index()
    )

    plt.style.use("bmh")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ordered_c1 = sorted(c1_values)
    sj_data = [df[df["c1"] == val]["s_j"].values for val in ordered_c1]
    axes[0].boxplot(sj_data, labels=[f"{v:.2f}" for v in ordered_c1], patch_artist=True)
    axes[0].set_title("Noise Effect on Failure Count $s_j$")
    axes[0].set_xlabel("c1")
    axes[0].set_ylabel("s_j")

    g = grouped.sort_values("c1")
    axes[1].plot(g["c1"], g["reject_rate"], marker="o", label="Reject rate")
    axes[1].plot(g["c1"], g["one_acc_rate"], marker="s", label="ONE_ACC rate")
    axes[1].set_title("Threshold Tuning under Noise")
    axes[1].set_xlabel("c1")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    summary = {
        "trials_per_c1": trials,
        "depolarizing_prob": depolarizing_prob,
        "readout_error_prob": readout_error_prob,
        "rows": int(len(df)),
        "by_c1": grouped.to_dict(orient="records"),
    }
    return {"summary": summary, "csv_path": str(csv_path), "plot_path": str(plot_path)}
