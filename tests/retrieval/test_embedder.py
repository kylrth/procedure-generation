import re
import unittest

import pytest

from retrieval import embedder


class TestBatch(unittest.TestCase):
    def test_one(self):
        batches = list(
            embedder.batch(
                [
                    [0] * 5,
                ],
                20,
            )
        )

        self.assertEqual(
            batches,
            [
                [[0] * 5],
            ],
        )

    def test_simple(self):
        batches = list(
            embedder.batch(
                [
                    [0] * 5,
                    [0] * 3,
                    [0] * 3,
                    [0] * 7,
                    [0] * 3,
                ],
                10,
            )
        )

        self.assertEqual(
            batches,
            [
                [
                    [0] * 5,
                    [0] * 3,
                ],
                [
                    [0] * 3,
                    [0] * 7,
                ],
                [
                    [0] * 3,
                ],
            ],
        )

    def test_too_long(self):
        batches = embedder.batch(
            [
                [0] * 5,
                [0] * 3,
                [0] * 3,
                [0] * 7,
                [0] * 3,
            ],
            6,
        )

        batches = iter(batches)
        next(batches)

        with pytest.raises(ValueError, match=re.escape("sequence of length 7 is too long (> 6)")):
            next(batches)


if __name__ == "__main__":
    unittest.main()
