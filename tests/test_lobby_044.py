from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_lobby_044 import (  # noqa: E402
    decode_fixed_state_features,
    empty_metrics,
    record_prediction,
    semantic_error,
    stratum_for_round,
    summarize_prediction_metrics,
    validate_rank,
)


def test_round_strata_boundaries():
    assert stratum_for_round(1) == "round_1_3"
    assert stratum_for_round(3) == "round_1_3"
    assert stratum_for_round(4) == "round_4_6"
    assert stratum_for_round(6) == "round_4_6"
    assert stratum_for_round(7) == "round_7_plus"
    assert stratum_for_round(99) == "round_7_plus"


def test_prediction_matrix_and_slot_semantics():
    metrics = empty_metrics()
    for label, pred in [(0, 0), (0, 1), (4, 5), (18, 20), (34, 36), (11, 1)]:
        record_prediction(metrics, label, pred)
    result = summarize_prediction_metrics(metrics)
    assert result["rows"] == 6
    assert result["correct"] == 1
    assert result["confusion_total"] == 6
    assert result["semantic_errors"]["missing_end_turn"] == 1
    assert result["semantic_errors"]["buy_slot_confusion"] == 1
    assert result["semantic_errors"]["play_slot_confusion"] == 1
    assert result["semantic_errors"]["discover_slot_confusion"] == 1
    assert result["semantic_errors"]["sell_as_reroll"] == 1
    assert semantic_error(5, 6) == "buy_slot_confusion"


def test_fixed_state_feature_decoding_counts_occupied_slots():
    obs = {
        "player": np.asarray([0, 0.05, 1, 0.7, 1.5, 0, 0, 0, 0], dtype=np.float32),
        "board": np.ones((7, 9), dtype=np.float32),
        "hand": np.ones((10, 9), dtype=np.float32),
    }
    obs["board"][:3, 0] = 0
    obs["hand"][:4, 0] = 0
    assert decode_fixed_state_features(obs) == {"gold": 7, "tier": 3, "board_count": 3, "hand_count": 4}


def test_rank_preserves_half_integer_and_rejects_quarter():
    assert validate_rank(7.5) == 7.5
    assert validate_rank(4) == 4.0
    try:
        validate_rank(7.25)
    except ValueError as exc:
        assert "half-integer" in str(exc)
    else:
        raise AssertionError("Expected quarter-rank rejection")
