"""Single versioned action registry shared by masks and execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    kind: str
    index: int | None = None


ACTIONS = tuple(
    [Action(kind) for kind in ("end", "reroll", "upgrade", "freeze")]
    + [
        Action(kind, index)
        for kind, count in (("buy", 7), ("sell", 7), ("play", 10), ("swap", 6), ("discover", 3))
        for index in range(count)
    ]
)
ACTION_COUNT = len(ACTIONS)
ACTION_VERSION = "discrete-37-v1"


def decode_action(value: int) -> Action:
    if type(value) is not int or not 0 <= value < ACTION_COUNT:
        raise ValueError(f"Action must be an integer in [0, {ACTION_COUNT}): {value!r}")
    return ACTIONS[value]


def encode_action(action: Action) -> int:
    if not isinstance(action, Action) or (
        action.index is not None and type(action.index) is not int
    ):
        raise ValueError(f"Invalid action: {action!r}")
    try:
        return ACTIONS.index(action)
    except ValueError as error:
        raise ValueError(f"Unknown action: {action!r}") from error
