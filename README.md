# Gottesman-Chuang Quantum Digital Signature (QDS) Simulator

This project implements a parameterized simulation framework for the Gottesman-Chuang Quantum Digital Signature (QDS) protocol. It provides classical scripts to model the quantum states, signing logic, and verification mechanisms mathematically required for QDS, along with tools to run experiments on forgery, repudiation, and noise sensitivity. Please note that this is a mathematical and experimental simulator designed for studying the quantum protocol logic; it is not a production-ready cryptographic system or an exact simulation of quantum network topology over optical links.

## High-Level Conceptual Overview

A Digital Signature scheme requires that messages can be signed by an author and verified by recipients such that the signature is unfalsifiable (forgery resistance) and non-repudiable (transferability).

In our QDS simulation, we have three main roles:
1. **Alice (Signer)**: Generates a set of classical private keys and computes their corresponding quantum public keys. She distributes the public keys to the recipients ahead of time.
2. **Recipients (Bob, Charlie...)**: Receive and store quantum public keys from Alice. During verification, they check the validity of the quantum signature against their stored copies.
3. **Distribution Center/Checker (Optional)**: Facilitates secure key-distribution to prevent Alice from sending mismatched keys to different recipients (which could cause a repudiation attack).

The protocol occurs in these phases:
1. **Key Generation**: Alice creates private key bits and generates the quantum one-way functions states $|f_k\rangle$.
2. **Distribution**: Alice sends the public quantum states to Bob and Charlie. (Copy limits apply).
3. **Signing**: Alice reveals the classical private key to sign a specific message bit.
4. **Verification**: The recipients check if the revealed classical key produces a state that overlaps sufficiently with their stored quantum state. The verification tracks the number of mismatch failures ($s_j$) across $M$ parallel tests.
5. **Decision**: Based on $s_j$ and two thresholds ($c_1$ and $c_2$), the recipient outputs one of three outcomes: `ONE_ACC` (transferably accepted), `ZERO_ACC` (weakly accepted), or `REJ` (rejected).

## Prerequisites

- **Python**: Version 3.8 or higher.
- **Operating System**: Cross-platform (Windows, macOS, Linux).
- **Core Dependencies**: `qiskit`, `qiskit-aer`, `numpy`, `argparse`, `pytest`, `matplotlib`.
- **Optional**: Jupyter Notebook for running interactive demos.

*Note: Simulation costs increase exponentially with the number of qubits ($n$). Keep $n$ small when running local statevector simulations.*

## Installation and Quickstart

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TSS99/Quantum_Signature.git
   cd Quantum_Signature/quantum_digital_signatures
   ```

2. **Set up a virtual environment and install**:
   ```bash
   python -m venv .venv
   # On Windows: .venv\Scripts\activate
   # On macOS/Linux: source .venv/bin/activate
   pip install -e .
   ```

3. **Run a quick demo**:
   ```bash
   python -m qds.cli demo quick
   ```

   **Expected Output Template**:
   ```text
   QDS Run Parameters
   ----------------------------------------------------
   M          : 32
   c1         : 0.0500
   c2         : 0.2000
   family     : angle1q
   n          : 1
   L          : 8
   T_max      : 4
   shots      : 1
   seed       : 42

   Recipient Verification
   ----------------------------------------------------
   recipient       s_j     outcome
   bob             0       VerificationOutcome.ONE_ACC
   charlie         0       VerificationOutcome.ONE_ACC

   Demo output written: artifacts/demo_quick.json
   ```
   Artifacts such as JSON bundles, experiment CSVs, and plots are saved in the `artifacts/` directory by default.

## Project Structure

- `src/qds/qowf/`: Contains the quantum one-way function definitions (`base.py`, `angle1q.py`, `fingerprint.py`).
- `src/qds/quantum/`: Quantum utilities and routines like the overlap calculation and swap test (`swap_test.py`, `verify.py`, `utils.py`).
- `src/qds/protocol/`: Core protocol logic for key distribution, signing, verification, and decision thresholds (`entities.py`, `distribution.py`, `signing.py`, `verification.py`, `outcomes.py`).
- `src/qds/experiments/`: High-level simulation scripts running targeted attacks and noise analysis (`forgery.py`, `repudiation.py`, `noise.py`).
- `src/qds/cli.py`: The command-line interface entry point.
- `tests/`: Contains `pytest` suites securing the math and threshold logic framework.
- `notebooks/`: Includes interactive Jupyter notebooks covering the protocol logic visually.

## Mathematical Intuitions and Steps

### 6.1 Quantum One-Way Function: $k \to |f_k\rangle$

The security of the QDS scheme rests on a Quantum One-Way Function (QOWF). The property we need is that preparing a state $|f_k\rangle$ given a classical key $k$ is easy, but guessing $k$ given a limited number of copies of $|f_k\rangle$ is mathematically hard.

This simulator implements two QOWF families:

**Family A: Single-qubit angle states (`angle1q`)**
- An $L$-bit key $k$ is mapped to an integer $j \in \{0, 1, \dots, 2^L - 1\}$.
- We define an angle step $\theta = \frac{\pi}{2^L}$.
- The state is generated as:
  $$ |f_k\rangle = \cos(j\theta)|0\rangle + \sin(j\theta)|1\rangle $$
- In the circuit, this state is easily prepared by applying a $R_y(2j\theta)$ gate to the $|0\rangle$ state, which directly matches the probability amplitudes.

**Family B: Small-$n$ fingerprint-style phase states (`fingerprint`)**
- Uses $n$ qubits, giving a dimension $N = 2^n$.
- We start with an equal superposition of all computational basis states $H^{\otimes n}|0\rangle^{\otimes n}$.
- A pseudo-random hash $h_k(x)$ derived from SHA256 expansion applies a phase to each basis state:
  $$ |f_k\rangle = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} (-1)^{h_k(x)} |x\rangle $$
- Computational-basis measurements yield uniform probabilities, completely destroying the phase information. Therefore, measuring $|f_k\rangle$ gives negligible intelligence about $k$.

### 6.2 Overlap as the Verification Primitive

Verification checks if the revealed classical key maps to the stored quantum state. Let $|\psi\rangle$ be the state stored by the recipient and $|f_k\rangle$ be the state re-generated using the revealed classical key $k$.
The probability of the verification succeeding is directly proportional to the state overlap $|\langle \psi | f_k \rangle|^2$.

We implement two mode-checks for this:
1. **Uncompute-and-measure**: Applies the inverse unitary $U_k^\dagger$ sequentially and measures the result. If they match completely, the state returns exactly to $|0\dots 0\rangle$.
2. **Swap Test**: A distinct mechanism that compares any two states without strictly requiring their preparation unitary.

### 6.3 Swap Test Mathematics

The swap test probabilistically compares two states $|\phi\rangle$ and $|\psi\rangle$ for equality.
1. The circuit sandwiches an ancilla qubit initialized in $|0\rangle$ between Hadamard gates.
2. A controlled-SWAP gate swaps the registers storing $|\phi\rangle$ and $|\psi\rangle$ conditioned on the ancilla.
3. The ancilla is measured.
4. The probability of measuring $0$ on the ancilla is:
   $$ P(\text{ancilla} = 0) = \frac{1 + |\langle\phi|\psi\rangle|^2}{2} $$

If $|\phi\rangle = |\psi\rangle$, the overlap is $1$, and $P(\text{ancilla} = 0) = 1$.
Because it is probabilistic, distinguishing somewhat similar states from identical ones may require multiple measurement shots. In exact simulation mode (`--exact`), we compute the statevector overlap explicitly to avoid sampling noise.

### 6.4 Thresholding and Outcomes ($c_1$, $c_2$)

Security against noise or malicious adversaries involves thresholding. Alice provides $M$ parallel public keys per signed bit. The recipient tallies failures or mismatches $s_j$ over the $M$ checks.

Decisions are formalized based on constants $c_1 < c_2$:
- **Transferable Accept (`ONE_ACC`)**: $s_j \le \lfloor c_1 M \rfloor$. The recipient accepts the message and is confident it can be forwarded to another participant who will also accept it.
- **Reject (`REJ`)**: $s_j \ge \lceil c_2 M \rceil$. The recipient rejects the signature as invalid.
- **Weak Accept (`ZERO_ACC`)**: Otherwise. The signature is accepted for personal use, but the recipient knows forwarding it might result in a rejection due to the narrow margin.

The buffer zone between $c_1$ and $c_2$ shields honest participants from small statistical noise and ensures a dishonest Alice cannot orchestrate a scenario where Bob reliably outputs `ONE_ACC` while Charlie outputs `REJ`. Example: with $M=64$, $c_1=0.05$, $c_2=0.20$, Bob accepts unconditionally at $\le 3$ errors and rejects at $\ge 13$ errors.

### 6.5 Copy Limits and the No-Cloning Intuition

The simulator enforces a public-key copy limit $T_{\text{max}}$. Real quantum mechanics dictates that unknown quantum states cannot be perfectly cloned (The No-Cloning Theorem). However, software statevectors *can* be copied infinitely in memory.
To maintain protocol reality in the simulator, we logically restrict Alice or the protocol from generating more than $T_{\text{max}}$ copies of any one public key. Requesting more than this limit triggers an error in the simulation software to prevent "cheating by simulator artifacts."

## Protocol Walkthrough

To sign a single binary bit $b$:

1. **Key Generation**: Alice randomly generates two lists of $M$ classical keys: $k_0^{1 \dots M}$ for signing bit $0$, and $k_1^{1 \dots M}$ for signing bit $1$.
2. **Distribution**: Alice constructs the public keys $|f_{k_0^i}\rangle$ and $|f_{k_1^i}\rangle$ and securely sends them to Bob and Charlie, adhering to $T_{\text{max}}$.
3. **Signing**: Alice wishes to sign bit $b$. She classically sends $b$ and the full signature array $(k_b^1, \dots, k_b^M)$ to Bob.
4. **Verification**: Bob attempts to verify the signature array. Using the claimed keys $k_b^i$, he compares them to the quantum public keys $|f_{k_b^i}\rangle$ he originally stored.
5. **Scoring**: Bob counts the failures $s_j$. Utilizing equations from section 6.4, Bob outputs `ONE_ACC`, `ZERO_ACC`, or `REJ`.
6. **Disposal**: Used quantum states collapse on measurement and must be discarded. Keys used to sign bits cannot be reused.

*Multi-bit messages repeat this whole process with fresh independent keys for each new bit. An optional repetition code can be used to vote and protect message integrity.*

## Key Distribution Options

Dishonest Alice might send orthogonal states to Bob and Charlie. To mitigate this repudiation attack, the distribution phase can run in two modes:
1. **Direct Distribution**: Alice simply sends copies down optical channels. No verification is done prior.
2. **Trusted Distributor (Distributed Swap-Test)**: The public states pass through a check prior to active use. In this framework, we simulate checking local copies using the `distributed_swap_test`. The distribution center performs swap tests across Bob's and Charlie's key sets. If tests fail, the center aborts key registration and discards the states.

A thorough swap test drastically decreases Alice's chance of successfully launching a repudiation attack without detection.

## Experiments and What They Show

### 9.1 Forgery Experiment

**What it answers**: Can an eavesdropper or malicious recipient (Eve), given $T$ legitimate quantum copies of Alice's public key, deduce enough of $k$ to forge a signature?
**Implementation**:
- `angle1q`: Eve uses optimal $T$-copy phase estimation or successive measurement to guess the parameter angle $j$.
- `fingerprint`: Eve repeatedly measures her bounded copies in the computational basis to build a profile, which is mathematically useless for recovering the $h_k(x)$ phases.
**Command**:
```bash
python -m qds.cli experiment forgery --trials 200 --M 64 --T 4 --family angle1q
```
**Interpretation**: The summary outputs Eve's overall forgery success rate (achieving a `ONE_ACC` verification against a recipient).

### 9.2 Repudiation and Transferability Experiment

**What it answers**: How significantly does the gap between $c_1$ and $c_2$ combined with distributed consistency checks prevent a malicious Alice from sending bad states to Bob and good states to Charlie?
**Implementation**: Malicious Alice is configured to purposefully hand Bob valid keys and Charlie random keys.
**Command (Direct)**:
```bash
python -m qds.cli experiment repudiation --trials 100 --M 64
```
**Command (Distributed Check)**:
```bash
python -m qds.cli experiment repudiation --trials 100 --M 64 --with-distributed-swap-test
```
**Interpretation**: The difference between the scripts highlights that without distributed verification, a small fraction of keys might disagree. Enabling `--with-distributed-swap-test` reduces repudiation rates drastically.

### 9.3 Noise Experiment

**What it answers**: How do standard quantum hardware errors impact the verification protocol and false-rejection rate?
**Implementation**: Instantiates `qiskit-aer` depolarizing and readout error models. Valid keys are passed through this noisy simulation.
**Command**:
```bash
python -m qds.cli experiment noise --trials 100 --c1-values 0.0,0.05,0.1 --depolarizing-prob 0.01
```
**Interpretation**: Demonstrates that strictly enforcing $c_1 = 0$ causes high false rejection rates under noise. Setting $c_1 > 0$ absorbs typical quantum operational errors into the validation threshold.

## CLI Reference

Run via `python -m qds.cli <command> [options]` or `qds <command> [options]`:

- `keygen`: Generate a JSON payload containing the simulated classical and quantum key routing material.
  - Options: `--M`, `--family`, `--L`, `--n`, `--T-max`, `--recipients`, `--seed`, `--output`
- `sign`: Execute key generation, distribution, and a 1-bit signature.
  - Options: `--bit`, `--M`, `--c1`, `--c2`, `--T-max`, `--family`, `--L`, `--n`, `--recipients`, `--verification-mode`, `--distribution-mode`, `--shots`, `--exact`, `--seed`, `--output`
- `verify`: Load a JSON bundle from `sign` and force verification manually.
  - Options: `--signed-message`, `--recipient`, `--shots`, `--exact`, `--seed`
- `demo quick`: Run an end-to-end 1-bit pre-configured mock protocol.
  - Options: `--seed`, `--output`
- `experiment forgery`: Execute the forgery attack routines.
  - Options: `--trials`, `--M`, `--T`, `--c1`, `--c2`, `--T-max`, `--family`, `--L`, `--n`, `--seed`, `--artifacts-dir`
- `experiment repudiation`: Execute repudiation attack trials.
  - Options: `--trials`, `--M`, `--c1`, `--c2`, `--T-max`, `--family`, `--L`, `--n`, `--with-distributed-swap-test`, `--seed`, `--artifacts-dir`
- `experiment noise`: Test thresholds against simulated circuit errors.
  - Options: `--trials`, `--M`, `--c1-values`, `--c2`, `--T-max`, `--family`, `--L`, `--n`, `--depolarizing-prob`, `--readout-error-prob`, `--seed`, `--artifacts-dir`

## Testing

Ensure code stability by executing the `pytest` test suite:
```bash
pytest tests/
```
The suite evaluates swap test logic behavior (formula consistency, probabilities of orthogonal vs parallel states) and verifies boundary conditions for the threshold decision rules algorithm.

## Notebook Demo

An interactive Jupyter environment allows you to step through matrix calculations smoothly.
1. Run `jupyter notebook notebooks/demo.ipynb`.
2. Step through the execution cells.
3. Observe mathematical progression and visual representations of verification gaps dynamically.

## Interactive Animation

Open `animation/qds_animation.html` in any browser (Chrome, Edge, Firefox) to view a visually rich, interactive walkthrough of the entire QDS protocol. The animation contains 7 navigable scenes covering protocol overview, key generation with a Bloch sphere visualization, quantum state distribution, signing, swap-test verification with threshold bars, a forgery attempt simulation, and a protocol summary. No server or build step required.

## Performance Notes

For the `fingerprint` phase family, simulation scales poorly with larger $n$ due to the statevector complexity size requiring $N=2^n$ dense complex values calculated at runtime.
Keep $n \le 5$ for iterative tasks. The default is $n=1$.

## Limitations and Non-Goals

1. **Not a Security Proof**: This simulator verifies execution behavior practically; it does not replace mathematically exhaustive cryptographic proofs.
2. **Adversary Constraint**: We strictly simulate limited operational spaces for Eve (like brute force phase observation), not full generic collective quantum-attack state discrimination algorithms.
3. **Copy-Limit is Software Enforced**: The logic tracks identical states. A real quantum attacker operates under physical hardware enforcement (no-cloning), but we approximate it in logic tracking loops.
4. **No Real Network Modeling**: Time-delay propagation over optic fibers or pulse loss generation rates are completely excluded.
5. **Parameter Choices**: Default parameters are illustrative and optimized for fast computational simulation execution rather than cryptographic grade minimum viable thresholds.

## References

- Gottesman, D., & Chuang, I. (2001). *Quantum Digital Signatures*. arXiv:quant-ph/0105032.
- Swap Test concept originally detailed spanning numerous quantum learning methodologies and state discriminator checks.
