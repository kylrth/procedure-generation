from typing import List, Optional, Tuple
import unittest

from langchain.llms.fake import FakeListLLM

from systems import Model


class MockModel:
    def __init__(self, tc: unittest.TestCase, expect: List[str], response: List[str]):
        self.tc = tc
        self.expect = expect
        self.model = Model(FakeListLLM(responses=response))

    def build_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        return self.model.build_completion_prompt(prompt, context, examples)

    def generate(self, s: str):
        # Here we're reaching inside the FakeListLLM in the Model to get the step number, so we know
        # which string to compare with.
        i = self.model.model.i

        self.tc.assertTrue(
            i < len(self.expect), f"received call {i+1}; expected only {len(self.expect)}"
        )
        self.tc.assertEqual(self.expect[i], s)

        return self.model.generate(s)
