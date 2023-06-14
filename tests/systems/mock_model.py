import sys
import unittest

from langchain.llms.fake import FakeListLLM

from systems import Model


class MockModel(Model):
    max_tokens = sys.maxsize

    def __init__(self, tc: unittest.TestCase, expect: list[str], response: list[str]):
        super().__init__(FakeListLLM(responses=response))
        self.tc = tc
        self.expect = expect

    def generate(self, s: str) -> list[str]:
        # Here we're reaching inside the FakeListLLM to get the step number, so we know which string
        # to compare with.
        i = self.model.i

        self.tc.assertTrue(
            i < len(self.expect), f"received call {i+1}; expected only {len(self.expect)}"
        )
        self.tc.assertEqual(self.expect[i], s)

        return super().generate(s)

    def agenerate(self, s: str) -> list[str]:
        raise NotImplementedError
