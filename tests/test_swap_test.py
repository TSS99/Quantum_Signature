from __future__ import annotations

import pytest
from qiskit.quantum_info import Statevector

from qds.quantum.swap_test import swap_test_pass_probability_exact, swap_test_pass_probability_sampled


def test_swap_test_exact_identical_states_pass_with_probability_one() -> None:
    state = Statevector([1.0, 0.0])
    assert swap_test_pass_probability_exact(state, state) == pytest.approx(1.0, abs=1e-12)


def test_swap_test_exact_orthogonal_states_pass_with_probability_half() -> None:
    s0 = Statevector([1.0, 0.0])
    s1 = Statevector([0.0, 1.0])
    assert swap_test_pass_probability_exact(s0, s1) == pytest.approx(0.5, abs=1e-12)


def test_swap_test_sampled_agrees_for_easy_case() -> None:
    s0 = Statevector([1.0, 0.0])
    prob, _counts = swap_test_pass_probability_sampled(s0, s0, shots=256, seed=123)
    assert prob > 0.90
