import unittest

from hearthstone_ai.actions import Action, decode_action, encode_action


class ActionTests(unittest.TestCase):
    def test_registry_roundtrip(self):
        for value in range(37):
            self.assertEqual(encode_action(decode_action(value)), value)
        self.assertEqual(decode_action(28), Action("swap", 0))
        self.assertEqual(decode_action(34), Action("discover", 0))

    def test_invalid_action_rejected(self):
        for value in (-1, 37, True, 1.5, "1"):
            with self.assertRaises(ValueError):
                decode_action(value)
        with self.assertRaises(ValueError):
            encode_action(Action("play", 10))
