import pytest

from hearthstone_ai.game import Game


def discounted_return(rewards, gamma):
    return sum((gamma**index) * reward for index, reward in enumerate(rewards))


def play_empty_board(seed, policy):
    game = Game(max_turns=8, max_actions=24)
    game.reset(seed)
    rewards = []
    combats = []
    actions = 0

    while not game.done:
        action = 0 if policy == "end_immediately" else 3
        result = game.step(action)
        rewards.append(result.reward)
        actions += 1
        if "combat_result" in result.info:
            combats.append(
                {
                    "combat_result": result.info["combat_result"],
                    "damage_taken": result.info["damage_taken"],
                    "damage_dealt": result.info["damage_dealt"],
                    "turn": result.info["turn"],
                }
            )

    return {
        "actions": actions,
        "combats": combats,
        "hp": game.hp,
        "reward": sum(rewards),
        "discounted_099": discounted_return(rewards, 0.99),
        "discounted_1": discounted_return(rewards, 1.0),
    }


@pytest.mark.parametrize("seed", [7, 17, 27])
def test_discounted_objective_rewards_stalling_identical_empty_board_combats(seed):
    end_now = play_empty_board(seed, "end_immediately")
    freeze_until_forced = play_empty_board(seed, "freeze_until_forced")

    assert end_now["combats"] == freeze_until_forced["combats"]
    assert end_now["reward"] == freeze_until_forced["reward"] == -8.0
    assert end_now["hp"] == freeze_until_forced["hp"] == 2
    assert end_now["actions"] == 8
    assert freeze_until_forced["actions"] == 192

    assert end_now["discounted_099"] == pytest.approx(-7.72553055720799)
    assert freeze_until_forced["discounted_099"] == pytest.approx(-3.165257839237666)
    assert freeze_until_forced["discounted_099"] > end_now["discounted_099"]

    assert end_now["discounted_1"] == freeze_until_forced["discounted_1"] == -8.0
