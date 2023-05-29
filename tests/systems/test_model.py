from typing import List
import unittest

from langchain.llms.fake import FakeListLLM
from langchain.schema import AIMessage, ChatGeneration, HumanMessage, LLMResult, SystemMessage

from systems.model import Model


class FakeChatModel:
    def __init__(self, text: str):
        self.text = text

    def generate(self, _: str) -> LLMResult:
        return LLMResult(
            generations=[[ChatGeneration(text=self.text, message=AIMessage(content=self.text))]]
        )


class TestModel(unittest.TestCase):
    """The basic tests don't really test the construction code, so we test those separately below."""

    def test_basic_completion(self):
        response = "Hi there, I'm a fake completion model."
        model = Model(FakeListLLM(responses=[response]), False)

        out = model("What could possibly go wrong?")[0]

        self.assertEqual(response, out)

    def test_basic_chat(self):
        response = "Hi there, I'm a fake chat model."
        model = Model(FakeChatModel(response), True)

        out = model("What could possibly go wrong?")[0]

        self.assertEqual(response, out)

    def test_completion_prompt(self):
        want = (
            "Please respond to the following as if you don't speak English.\n\n"
            "===BEGIN EXAMPLE===\n"
            "What's your name?\n"
            "Désolé, je ne parle pas anglais.\n"
            "===END EXAMPLE===\n\n"
            "Hello\n"
        )

        got = Model(FakeListLLM(responses=[])).build_completion_prompt(
            "Hello",
            "Please respond to the following as if you don't speak English.",
            [("What's your name?", "Désolé, je ne parle pas anglais.")],
        )

        self.assertEqual(want, got)

    def test_chat_prompt(self):
        want = [
            SystemMessage(content="Please respond to the following as if you don't speak English."),
            HumanMessage(content="What's your name?"),
            AIMessage(content="Désolé, je ne parle pas anglais."),
            HumanMessage(content="Hello"),
        ]

        got = Model.build_chat_prompt(
            "Hello",
            "Please respond to the following as if you don't speak English.",
            [("What's your name?", "Désolé, je ne parle pas anglais.")],
        )

        self.assertEqual(want, got)


if __name__ == "__main__":
    unittest.main()
