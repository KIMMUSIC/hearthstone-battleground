"""Policies consume only their private view plus public lobby information."""


def strength(minion):
    return minion.attack + minion.health + 2 * minion.divine_shield + minion.taunt


def heuristic_policy(view):
    legal = view.legal_actions
    if not legal:
        raise ValueError("Policy called without legal actions")
    if view.pending_discover:
        return max(legal, key=lambda action: strength(view.pending_discover[action - 34]))
    plays = [a for a in legal if 18 <= a < 28]
    if plays:
        return max(plays, key=lambda action: strength(view.hand[action - 18]))
    if len(view.board) == 7 and view.hand:
        weakest = min(range(7), key=lambda i: strength(view.board[i]))
        if max(strength(m) for m in view.hand) > strength(view.board[weakest]):
            return 11 + weakest
    if 2 in legal and len(view.board) >= 3:
        return 2
    buys = [a for a in legal if 4 <= a < 11]
    if buys and len(view.hand) < 3:
        cards = {card.dbf_id: card for card in view.cards}

        def value(action):
            minion = view.shop[action - 4]
            copies = sum(m.card_id == minion.card_id and not m.golden
                         and cards[m.card_id].triple_allowed for m in view.board + view.hand)
            return strength(minion) + 3 * copies

        best = max(buys, key=value)
        if len(view.board) < 7 or value(best) > min(strength(m) for m in view.board):
            return best
    if 1 in legal and view.gold >= 4:
        return 1
    return 0


def random_policy(view, rng):
    if not view.legal_actions:
        raise ValueError("Policy called without legal actions")
    return rng.choice(view.legal_actions)
