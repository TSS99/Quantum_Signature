from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qds.experiments.forgery import run_forgery_experiment
from qds.experiments.noise import run_noise_experiment
from qds.experiments.repudiation import run_repudiation_experiment
from qds.protocol.entities import Recipient, SignedMessage
from qds.protocol.signing import ProtocolConfig, generate_keygen_bundle, run_single_bit_round
from qds.qowf import build_qowf
from qds.quantum.utils import ensure_artifacts_dir, seeded_rng


def _recipient_names(csv: str) -> list[str]:
    names = [item.strip() for item in csv.split(",") if item.strip()]
    if not names:
        raise ValueError("At least one recipient is required.")
    return names


def _dump_json(data: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_json_arg(value: str) -> dict[str, Any]:
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _print_run_header(config: ProtocolConfig) -> None:
    print("\nQDS Run Parameters")
    print("-" * 52)
    print(f"M          : {config.M}")
    print(f"c1         : {config.c1:.4f}")
    print(f"c2         : {config.c2:.4f}")
    print(f"family     : {config.family}")
    print(f"n          : {config.n}")
    print(f"L          : {config.L}")
    print(f"T_max      : {config.T_max}")
    print(f"shots      : {'exact' if config.shots is None else config.shots}")
    print(f"seed       : {config.seed}")


def _print_recipient_results(round_result: dict[str, Any]) -> None:
    print("\nRecipient Verification")
    print("-" * 52)
    print(f"{'recipient':<16}{'s_j':<8}{'outcome'}")
    for row in round_result.get("recipient_results", []):
        print(f"{row['recipient']:<16}{row['s_j']:<8}{row['outcome']}")


def cmd_keygen(args: argparse.Namespace) -> None:
    config = ProtocolConfig(
        M=args.M,
        T_max=args.T_max,
        L=args.L,
        n=args.n,
        family=args.family,
        seed=args.seed,
    )
    _print_run_header(config)
    bundle = generate_keygen_bundle(config=config, recipient_names=_recipient_names(args.recipients))
    _dump_json(bundle, Path(args.output))
    print(f"\nKey bundle written: {args.output}")


def cmd_sign(args: argparse.Namespace) -> None:
    config = ProtocolConfig(
        M=args.M,
        c1=args.c1,
        c2=args.c2,
        T_max=args.T_max,
        L=args.L,
        n=args.n,
        family=args.family,
        verification_mode=args.verification_mode,
        shots=None if args.exact else args.shots,
        seed=args.seed,
    )
    _print_run_header(config)
    out = run_single_bit_round(
        bit=args.bit,
        config=config,
        recipient_names=_recipient_names(args.recipients),
        distribution_mode=args.distribution_mode,
    )
    _print_recipient_results(out)
    _dump_json(out, Path(args.output))
    print(f"\nSigned message bundle written: {args.output}")


def cmd_verify(args: argparse.Namespace) -> None:
    payload = _load_json_arg(args.signed_message)
    params = payload["params"]
    config = ProtocolConfig(
        M=int(params["M"]),
        c1=float(params["c1"]),
        c2=float(params["c2"]),
        T_max=int(params["T_max"]),
        L=int(params["L"]),
        n=int(params["n"]),
        family=str(params["family"]),
        verification_mode=str(params["verification_mode"]),
        shots=None if args.exact else args.shots,
        seed=args.seed if args.seed is not None else params.get("seed"),
    )
    _print_run_header(config)

    recipient_keys = payload.get("recipient_public_keys", {}).get(args.recipient)
    if recipient_keys is None:
        raise ValueError(f"Recipient '{args.recipient}' not found in signed-message bundle.")

    qowf = build_qowf(family=config.family, L=config.L, n=config.n)
    verifier = Recipient(
        name=args.recipient,
        qowf=qowf,
        M=config.M,
        rng=seeded_rng(config.seed),
    )
    verifier.load_public_key_labels(recipient_keys)
    signed = SignedMessage.from_dict(payload["signed_message"])
    result = verifier.verify_signature(
        signed_message=signed,
        c1=config.c1,
        c2=config.c2,
        verification_mode=config.verification_mode,
        shots=config.shots,
    )
    print("\nVerification Result")
    print("-" * 52)
    print(f"recipient   : {result.recipient}")
    print(f"s_j         : {result.failures}")
    print(f"outcome     : {result.outcome.value}")


def cmd_demo_quick(args: argparse.Namespace) -> None:
    rng = seeded_rng(args.seed)
    bit = int(rng.integers(0, 2))
    config = ProtocolConfig(
        M=32,
        c1=0.05,
        c2=0.20,
        T_max=4,
        L=8,
        n=1,
        family="angle1q",
        verification_mode="uncompute",
        shots=1,
        seed=args.seed,
    )
    _print_run_header(config)
    out = run_single_bit_round(
        bit=bit,
        config=config,
        recipient_names=["bob", "charlie"],
        distribution_mode="direct",
    )
    _print_recipient_results(out)
    _dump_json(out, Path(args.output))
    print(f"\nDemo output written: {args.output}")


def cmd_experiment_forgery(args: argparse.Namespace) -> None:
    out = run_forgery_experiment(
        trials=args.trials,
        M=args.M,
        T=args.T,
        c1=args.c1,
        c2=args.c2,
        T_max=args.T_max,
        family=args.family,
        L=args.L,
        n=args.n,
        seed=args.seed,
        artifacts_dir=args.artifacts_dir,
    )
    print("\nForgery Experiment Summary")
    print("-" * 52)
    for k, v in out["summary"].items():
        print(f"{k:<40}{v}")
    print(f"CSV  : {out['csv_path']}")
    print(f"Plot : {out['plot_path']}")


def cmd_experiment_repudiation(args: argparse.Namespace) -> None:
    out = run_repudiation_experiment(
        trials=args.trials,
        M=args.M,
        c1=args.c1,
        c2=args.c2,
        T_max=args.T_max,
        family=args.family,
        L=args.L,
        n=args.n,
        with_distributed_swap_test=args.with_distributed_swap_test,
        seed=args.seed,
        artifacts_dir=args.artifacts_dir,
    )
    print("\nRepudiation Experiment Summary")
    print("-" * 52)
    for k, v in out["summary"].items():
        print(f"{k:<40}{v}")
    print(f"CSV  : {out['csv_path']}")
    print(f"Plot : {out['plot_path']}")


def cmd_experiment_noise(args: argparse.Namespace) -> None:
    c1_values = [float(x.strip()) for x in args.c1_values.split(",") if x.strip()]
    out = run_noise_experiment(
        trials=args.trials,
        M=args.M,
        c1_values=c1_values,
        c2=args.c2,
        T_max=args.T_max,
        family=args.family,
        L=args.L,
        n=args.n,
        depolarizing_prob=args.depolarizing_prob,
        readout_error_prob=args.readout_error_prob,
        seed=args.seed,
        artifacts_dir=args.artifacts_dir,
    )
    print("\nNoise Experiment Summary")
    print("-" * 52)
    summary = out["summary"]
    print(f"trials_per_c1          {summary['trials_per_c1']}")
    print(f"depolarizing_prob      {summary['depolarizing_prob']}")
    print(f"readout_error_prob     {summary['readout_error_prob']}")
    print(f"rows                   {summary['rows']}")
    print(f"CSV  : {out['csv_path']}")
    print(f"Plot : {out['plot_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qds",
        description="Quantum Digital Signature (Gottesman-Chuang) protocol simulator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_keygen = sub.add_parser("keygen", help="Generate key material bundle.")
    p_keygen.add_argument("--M", type=int, default=64)
    p_keygen.add_argument("--family", type=str, default="angle1q")
    p_keygen.add_argument("--L", type=int, default=8)
    p_keygen.add_argument("--n", type=int, default=1)
    p_keygen.add_argument("--T-max", type=int, default=4, dest="T_max")
    p_keygen.add_argument("--recipients", type=str, default="bob,charlie")
    p_keygen.add_argument("--seed", type=int, default=7)
    p_keygen.add_argument("--output", type=str, default="artifacts/keygen_bundle.json")
    p_keygen.set_defaults(func=cmd_keygen)

    p_sign = sub.add_parser("sign", help="Run one-bit signing and verification simulation.")
    p_sign.add_argument("--bit", type=int, required=True)
    p_sign.add_argument("--M", type=int, default=64)
    p_sign.add_argument("--c1", type=float, default=0.05)
    p_sign.add_argument("--c2", type=float, default=0.20)
    p_sign.add_argument("--T-max", type=int, default=4, dest="T_max")
    p_sign.add_argument("--family", type=str, default="angle1q")
    p_sign.add_argument("--L", type=int, default=8)
    p_sign.add_argument("--n", type=int, default=1)
    p_sign.add_argument("--recipients", type=str, default="bob,charlie")
    p_sign.add_argument("--verification-mode", type=str, default="uncompute")
    p_sign.add_argument("--distribution-mode", type=str, default="direct")
    p_sign.add_argument("--shots", type=int, default=1)
    p_sign.add_argument("--exact", action="store_true")
    p_sign.add_argument("--seed", type=int, default=7)
    p_sign.add_argument("--output", type=str, default="artifacts/signed_message.json")
    p_sign.set_defaults(func=cmd_sign)

    p_verify = sub.add_parser("verify", help="Verify a signed-message bundle for one recipient.")
    p_verify.add_argument("--signed-message", type=str, required=True)
    p_verify.add_argument("--recipient", type=str, default="bob")
    p_verify.add_argument("--shots", type=int, default=1)
    p_verify.add_argument("--exact", action="store_true")
    p_verify.add_argument("--seed", type=int, default=7)
    p_verify.set_defaults(func=cmd_verify)

    p_demo = sub.add_parser("demo", help="Demo routines.")
    demo_sub = p_demo.add_subparsers(dest="demo_command", required=True)
    p_demo_quick = demo_sub.add_parser("quick", help="Quick end-to-end demo.")
    p_demo_quick.add_argument("--seed", type=int, default=42)
    p_demo_quick.add_argument("--output", type=str, default="artifacts/demo_quick.json")
    p_demo_quick.set_defaults(func=cmd_demo_quick)

    p_exp = sub.add_parser("experiment", help="Attack/noise experiments.")
    exp_sub = p_exp.add_subparsers(dest="experiment_command", required=True)

    p_forgery = exp_sub.add_parser("forgery", help="Run forgery attack experiment.")
    p_forgery.add_argument("--trials", type=int, default=200)
    p_forgery.add_argument("--M", type=int, default=64)
    p_forgery.add_argument("--T", type=int, default=4)
    p_forgery.add_argument("--c1", type=float, default=0.05)
    p_forgery.add_argument("--c2", type=float, default=0.20)
    p_forgery.add_argument("--T-max", type=int, default=5, dest="T_max")
    p_forgery.add_argument("--family", type=str, default="angle1q")
    p_forgery.add_argument("--L", type=int, default=8)
    p_forgery.add_argument("--n", type=int, default=1)
    p_forgery.add_argument("--seed", type=int, default=7)
    p_forgery.add_argument("--artifacts-dir", type=str, default="artifacts")
    p_forgery.set_defaults(func=cmd_experiment_forgery)

    p_rep = exp_sub.add_parser("repudiation", help="Run repudiation/transferability experiment.")
    p_rep.add_argument("--trials", type=int, default=200)
    p_rep.add_argument("--M", type=int, default=64)
    p_rep.add_argument("--c1", type=float, default=0.05)
    p_rep.add_argument("--c2", type=float, default=0.20)
    p_rep.add_argument("--T-max", type=int, default=4, dest="T_max")
    p_rep.add_argument("--family", type=str, default="angle1q")
    p_rep.add_argument("--L", type=int, default=8)
    p_rep.add_argument("--n", type=int, default=1)
    p_rep.add_argument("--with-distributed-swap-test", action="store_true")
    p_rep.add_argument("--seed", type=int, default=19)
    p_rep.add_argument("--artifacts-dir", type=str, default="artifacts")
    p_rep.set_defaults(func=cmd_experiment_repudiation)

    p_noise = exp_sub.add_parser("noise", help="Run noise-threshold experiment.")
    p_noise.add_argument("--trials", type=int, default=200)
    p_noise.add_argument("--M", type=int, default=64)
    p_noise.add_argument("--c1-values", type=str, default="0.0,0.05,0.1")
    p_noise.add_argument("--c2", type=float, default=0.20)
    p_noise.add_argument("--T-max", type=int, default=4, dest="T_max")
    p_noise.add_argument("--family", type=str, default="angle1q")
    p_noise.add_argument("--L", type=int, default=8)
    p_noise.add_argument("--n", type=int, default=1)
    p_noise.add_argument("--depolarizing-prob", type=float, default=0.01)
    p_noise.add_argument("--readout-error-prob", type=float, default=0.02)
    p_noise.add_argument("--seed", type=int, default=11)
    p_noise.add_argument("--artifacts-dir", type=str, default="artifacts")
    p_noise.set_defaults(func=cmd_experiment_noise)

    p_alias_f = sub.add_parser("run-forgery-experiment", help="Alias for experiment forgery.")
    p_alias_f.add_argument("--trials", type=int, default=200)
    p_alias_f.add_argument("--M", type=int, default=64)
    p_alias_f.add_argument("--T", type=int, default=4)
    p_alias_f.add_argument("--seed", type=int, default=7)
    p_alias_f.set_defaults(
        func=lambda a: cmd_experiment_forgery(
            argparse.Namespace(
                trials=a.trials,
                M=a.M,
                T=a.T,
                c1=0.05,
                c2=0.20,
                T_max=5,
                family="angle1q",
                L=8,
                n=1,
                seed=a.seed,
                artifacts_dir="artifacts",
            )
        )
    )

    p_alias_t = sub.add_parser("run-transferability-experiment", help="Alias for experiment repudiation.")
    p_alias_t.add_argument("--trials", type=int, default=200)
    p_alias_t.add_argument("--M", type=int, default=64)
    p_alias_t.add_argument("--with-distributed-swap-test", action="store_true")
    p_alias_t.add_argument("--seed", type=int, default=19)
    p_alias_t.set_defaults(
        func=lambda a: cmd_experiment_repudiation(
            argparse.Namespace(
                trials=a.trials,
                M=a.M,
                c1=0.05,
                c2=0.20,
                T_max=4,
                family="angle1q",
                L=8,
                n=1,
                with_distributed_swap_test=a.with_distributed_swap_test,
                seed=a.seed,
                artifacts_dir="artifacts",
            )
        )
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    ensure_artifacts_dir("artifacts")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
