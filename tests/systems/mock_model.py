import sys
import unittest

from langchain_community.llms.fake import FakeListLLM
from langchain_core.messages import BaseMessage

from systems import Model


class MockModel(Model):
    max_tokens = sys.maxsize

    def __init__(self, tc: unittest.TestCase, expect: list[str], response: list[str]):
        super().__init__(FakeListLLM(responses=response))
        self.tc = tc
        self.expect = expect

    async def generate(self, final_prompt: str | list[BaseMessage]) -> str:
        # Here we're reaching inside the FakeListLLM to get the step number, so we know which string
        # to compare with.
        i = self.model.i

        self.tc.assertTrue(
            i < len(self.expect), f"received call {i+1}; expected only {len(self.expect)}"
        )
        self.tc.assertEqual(self.expect[i], final_prompt)

        return await super().generate(final_prompt)

    def agenerate(self, s: str) -> list[str]:
        raise NotImplementedError
