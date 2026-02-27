# Quantum Digital Signatures Simulator

A runnable Python simulator for the Gottesman-Chuang quantum digital signature (QDS) idea.

Important: this project is a **protocol simulator**, not a claim of real-world cryptographic security.
It models unknown-state behavior using statevectors and enforces no-cloning only at simulator logic level (copy limits, single-use rules).

## What is implemented

- Quantum one-way function (QOWF) interface `QuantumOneWayFunction` with:
  - `prepare_circuit(k)`
  - `statevector(k)`
  - `overlap(k1, k2)`
- QOWF Family 1 (`angle1q`):
  - `|f_k> = cos(j*theta)|0> + sin(j*theta)|1>` with `theta = pi/2^L`
  - Circuit implementation via `Ry(2*j*theta)`
- QOWF Family 2 (`fingerprint`, `n in {3,4,5}`):
  - Uniform superposition + hash-derived phase signs
  - Circuit path + fast direct statevector path
- Swap test:
  - Exact pass probability (statevector)
  - Sampled pass probability (Aer qasm, optional noise)
  - Single-shot mode
- Verification:
  - Exact overlap acceptance `|<psi|f_k>|^2`
  - Circuit uncompute-and-measure implementation
- Protocol simulator:
  - Alice/Bob/Charlie entities
  - One-bit signing and verification with outcomes:
    - `ONE_ACC`
    - `ZERO_ACC`
    - `REJ`
  - Threshold decision with `c1 < c2`
  - Copy limit enforcement (`T_max`)
  - Single-use key enforcement
  - Multi-bit repetition mode (`run_multibit_repetition`)
- Key distribution modes:
  - Direct distribution
  - Trusted distributor with swap-test checking
  - Distributed swap-test mode for two recipients
- Experiments:
  - Forgery attack (angle estimator + naive fingerprint attack)
  - Repudiation/transferability (dishonest Alice mismatch strategy)
  - Noise experiment (depolarizing + readout error; threshold tuning)
- CLI commands and artifact outputs (`CSV + PNG` under `artifacts/`)
- Unit tests for swap test, verification, and thresholds/single-use/copy-limit logic

## Project structure

```text
quantum_digital_signatures/
  pyproject.toml
  README.md
  src/qds/
    __init__.py
    qowf/
      __init__.py
      base.py
      angle1q.py
      fingerprint.py
    quantum/
      __init__.py
      swap_test.py
      verify.py
      utils.py
    protocol/
      __init__.py
      entities.py
      distribution.py
      signing.py
      verification.py
      outcomes.py
    experiments/
      __init__.py
      forgery.py
      repudiation.py
      noise.py
    cli.py
  tests/
    conftest.py
    test_swap_test.py
    test_verify.py
    test_protocol.py
  notebooks/
    demo.ipynb
  artifacts/
```

## Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
```

If your environment blocks bytecode writes, run commands with:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
```

## CLI usage

You can use either:

- `python -m qds.cli ...`
- `qds ...` (after editable install)

### Required-style demo commands

```powershell
python -m qds.cli demo quick
python -m qds.cli sign --bit 1 --M 64 --family angle1q
python -m qds.cli verify --signed-message artifacts/signed_message.json --recipient bob
python -m qds.cli experiment forgery --trials 200 --M 64 --T 4
python -m qds.cli experiment repudiation --trials 200 --with-distributed-swap-test
```

### Additional commands

```powershell
python -m qds.cli keygen --M 64 --family fingerprint --n 4
python -m qds.cli experiment noise --trials 200 --M 64 --c1-values 0.0,0.05,0.1
python -m qds.cli run-forgery-experiment --trials 200 --M 64 --T 4
python -m qds.cli run-transferability-experiment --trials 200 --with-distributed-swap-test
```

## Output fields per run

Each protocol run prints:

- `M`, `c1`, `c2`, `family`, `n`, `L`, `T_max`, `shots`, `seed`
- For each recipient: `s_j` and outcome label (`ONE_ACC`, `ZERO_ACC`, `REJ`)

Experiments write:

- CSV summary table in `artifacts/`
- PNG plot in `artifacts/`

## Parameter meanings

- `M`: number of key checks per signed bit
- `c1`, `c2`: decision thresholds (`c1 < c2`)
- `T_max`: max copies in circulation per key (enforced)
- `L`: private key length in bits
- `n`: qubit count for family (`angle1q` uses `n=1`; fingerprint uses `3/4/5`)
- `shots`: sampled circuit evaluations per key-check (`1` gives per-key Bernoulli checks)
- `seed`: deterministic RNG seed for reproducibility

## Experiments summary

- Forgery:
  - Angle family uses mixed `X/Z` measurement estimator for key index
  - Fingerprint family uses naive computational-basis attack (fails to recover phases)
- Repudiation/transferability:
  - Dishonest Alice sends mismatched public keys to Bob/Charlie
  - Distributed swap-test mode sharply suppresses successful disagreement (often via abort)
- Noise:
  - Adds depolarizing and readout error in Aer
  - Demonstrates effect of nonzero `c1` on tolerance to noise-induced failures

## Tests

```powershell
python -m pytest -q
```

Current test scope:

- Swap-test exact + sampled behavior
- Verification exact behavior
- Threshold decision regions
- Copy-limit and single-use enforcement

## Notebook

Open `notebooks/demo.ipynb`.

It demonstrates:

- accept/reject outcomes versus `M`
- noise impact on `s_j` distribution and outcome rates

## Limitations

- No unconditional security proof; empirical simulation only.
- Public keys are simulated with full statevector access for implementation convenience.
- No physical no-cloning: copy limits are enforced at protocol logic level.
- Noise model is basic and gate-level, not hardware-calibrated.
- Fingerprint circuit currently uses a diagonal-phase construction suitable for small `n` demos.
