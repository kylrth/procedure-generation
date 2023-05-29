from typing import List

from langchain.prompts.example_selector.base import BaseExampleSelector

from .interface import SystemInterface
from .model import Model


def format_recipe(ingredients: List[str], directions: List[str]) -> str:
    return "\n\n".join(
        (
            "\n".join("- " + ingredient for ingredient in ingredients),
            "\n".join(f"{i+1}. {step}" for i, step in enumerate(directions)),
        )
    )


class FewShot(SystemInterface):
    """This model prompts an LM to generate text few-shot, with the examples provided by searching a
    vector store for texts with similar embeddings to the title."""

    def __init__(self, model: Model, instructions: str, selector: BaseExampleSelector):
        self.model = model
        self.instructions = instructions
        self.selector = selector

    def generate(self, title: str) -> List[str]:
        examples = self.selector.select_examples({"title": title})

        print(examples)
        examples = [(example["title"], example["recipe"]) for example in examples]

        return self.model(title, self.instructions, examples)
