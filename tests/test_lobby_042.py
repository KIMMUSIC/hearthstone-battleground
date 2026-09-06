from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_lobby_042 import classify_semantic_error, common_dev_errors, summarize_confusion, validate_rank  # noqa: E402


def test_confusion_totals_and_macro_accuracy():
    labels = np.asarray([0, 0, 1, 4, 5, 18, 18, 34], dtype=np.int64)
    predictions = np.asarray([0, 1, 0, 5, 5, 0, 19, 36], dtype=np.int64)
    result = summarize_confusion(labels, predictions)
    assert result["confusion_total"] == len(labels)
    assert sum(sum(row) for row in result["confusion_matrix"]) == len(labels)
    assert result["per_action"]["0"]["count"] == 2
    assert result["per_action"]["0"]["correct"] == 1
    assert result["per_action"]["5"]["accuracy"] == 1.0
    assert result["macro_action_accuracy"] == (0.5 + 0.0 + 0.0 + 1.0 + 0.0 + 0.0) / 6


def test_semantic_error_grouping_names_key_mistakes():
    assert classify_semantic_error(0, 1) == "missing_end_turn"
    assert classify_semantic_error(18, 0) == "early_end_turn"
    assert classify_semantic_error(4, 5) == "buy_slot_confusion"
    assert classify_semantic_error(18, 19) == "play_slot_confusion"
    assert classify_semantic_error(34, 36) == "discover_slot_confusion"
    assert classify_semantic_error(2, 4) == "upgrade_as_buy"


def test_common_dev_errors_counts_union_and_intersection_rows():
    labels = np.asarray([0, 0, 4, 18, 18], dtype=np.int64)
    predictions = {
        7: [0, 1, 5, 18, 0],
        17: [0, 2, 4, 19, 0],
        27: [0, 3, 5, 20, 0],
    }
    result = common_dev_errors(labels, predictions)
    assert result["rows"] == 5
    assert result["union_wrong_rows"] == 4
    assert result["intersection_wrong_rows"] == 2
    assert result["per_label_intersection_wrong_rows"] == {"0": 1, "18": 1}
    assert result["common_error_prediction_instances"]["confusion_total"] == 6


def test_validate_rank_preserves_half_integer_and_rejects_quarter_rank():
    assert validate_rank(7.5) == 7.5
    assert validate_rank(4) == 4.0
    try:
        validate_rank(7.25)
    except ValueError as exc:
        assert "half-integer" in str(exc)
    else:
        raise AssertionError("Expected quarter-rank rejection")
