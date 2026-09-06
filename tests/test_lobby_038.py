"""Hand calculated episode and rollout boundaries for direct reward attribution."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_lobby_038 import credit  # noqa: E402


def episode(end, rank=8):
    return {"num_timesteps": end, "rank": rank, "truncated": False}


def test_rollout_boundary_and_unfinished_tail():
    result = credit([episode(4), episode(6)], 10, 4, 1, .5)
    assert result["same_rollout_terminal_coverage"] == .6
    assert result["unfinished_tail_steps"] == 4
    assert result["mean_absolute_direct_terminal_reward_component"] == pytest.approx((1+.5+.25+.125+1+.5)/10)


def test_terminal_on_next_rollout_does_not_cross_boundary():
    result = credit([episode(5)], 8, 4, 1, 1)
    assert result["same_rollout_terminal_coverage"] == 1/8
    assert result["mean_absolute_direct_terminal_reward_component"] == 1/8


def test_zero_reward_episode_and_invalid_order():
    result = credit([episode(2, 4.5), episode(3)], 4, 4, 1, 1)
    assert result["same_rollout_terminal_coverage"] == 3/4
    assert result["nonzero_direct_reward_coverage"] == 1/4
    with pytest.raises(ValueError):
        credit([episode(3), episode(2)], 4, 4, 1, 1)
