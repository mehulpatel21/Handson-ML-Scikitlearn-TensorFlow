import unittest
from src.math import Math

class MathTest(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(Math.addition(4,4), 8)


