from collections.abc import Callable
from dataclasses import dataclass

from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from dataset import Doc


@dataclass
class ModelDetails:
    is_chat: bool
    max_tokens: int
    langchain_model: Callable[..., BaseLanguageModel]


# enumerates the models that we can use
model_info: dict[str, dict[str, ModelDetails]] = {
    "openai": {
        "gpt-3.5-turbo-0613": ModelDetails(True, 4096, ChatOpenAI),
        "gpt-3.5-turbo-16k-0613": ModelDetails(True, 16384, ChatOpenAI),
        "gpt-4-0613": ModelDetails(True, 8192, ChatOpenAI),
        "gpt-4-32k-0613": ModelDetails(True, 32768, ChatOpenAI),
        "text-davinci-003": ModelDetails(False, 4097, OpenAI),
        "text-davinci-002": ModelDetails(False, 4097, OpenAI),
        "text-curie-001": ModelDetails(False, 2049, OpenAI),
        "text-babbage-001": ModelDetails(False, 2049, OpenAI),
        "text-ada-001": ModelDetails(False, 2049, OpenAI),
    }
}


class Model:
    """Model provides an abstraction around chat and text completion models suitable for this
    project, so that we can use chat and completion models with the same code."""

    # This is the string used to format examples into the prompts for completion models.
    example_fmt: str = "===BEGIN EXAMPLE===\n{input}\n\n{output}\n===END EXAMPLE==="

    # the number of tokens this model supports; set automatically by from_full_name
    max_tokens: int | None = None

    name: str

    def __init__(self, model: BaseLanguageModel, chat: bool = False):
        self.model = model
        self.chat = chat

    @classmethod
    def from_full_name(cls, full_name: str, **kwargs) -> "Model":
        """Create a new model from a full name, which includes the service and model name.

        The full_name parameter combines the name of an LLM service (e.g. "openai") with the name of
        a model (e.g. "gpt-3.5-turbo"), separated by a hyphen (e.g. "openai-gpt-3.5-turbo").

        All kwargs are passed to the langchain model constructor.
        """
        service, model = full_name.lower().split("-", 1)

        try:
            details = model_info[service][model]
        except KeyError:
            raise NotImplementedError(full_name) from None

        model = details.langchain_model(model=model, **kwargs)

        out = cls(model, details.is_chat)
        out.max_tokens = details.max_tokens
        out.name = full_name

        return out

    def __call__(
        self,
        prompt: str,
        context: str | None = None,
        examples: list[Doc] | None = None,
    ) -> list[str]:
        """Generate calls the model with the prompt and returns the response text.

        If context is provided, it is added before the prompt. If example are provided, they are
        added between the context and the prompt. For a text completion model, the full prompt looks
        like this:

            {context}

            ===BEGIN EXAMPLE===
            {examples[0][0]}

            {examples[0][1]}
            ===END EXAMPLE===

            ===BEGIN EXAMPLE===
            {examples[1][0]}

            {examples[1][1]}
            ===END EXAMPLE===

            ...

            {prompt}

        For a chat model, the messages look like this:

            System: {context}
            Human: {examples[0][0]}
            Model: {examples[0][1]}
            Human: {examples[1][0]}
            Model: {examples[1][1]}
            ...
            Human: {prompt}
        """
        full_prompt = self.build_prompt(prompt, context, examples)
        return self.generate(full_prompt)

    def generate(self, final_prompt: str | list[BaseMessage]) -> list[str]:
        """Calls the model on the already-constructed prompt. Can be used in conjunction with
        build_prompt to re-use constructed prompts."""
        out = self.model.generate([final_prompt])

        # In the future we may surface other information from the results, but for now we just need
        # the strings.
        return [r.text for r in out.generations[0]]

    async def agenerate(self, final_prompt: str | list[BaseMessage]) -> list[str]:
        out = await self.model.agenerate([final_prompt])

        # In the future we may surface other information from the results, but for now we just need
        # the strings.
        return [r.text for r in out.generations[0]]

    def build_prompt(
        self,
        prompt: str,
        context: str | None = None,
        examples: list[Doc] | None = None,
    ) -> str | list[BaseMessage]:
        """Builds the final prompt as described in documentation for __call__, but returns the
        constructed prompt instead of running the model.

        The type of the output depends on whether this is a chat or completion model."""
        if self.chat:
            return self.build_chat_prompt(prompt, context, examples)

        return self.build_completion_prompt(prompt, context, examples)

    @staticmethod
    def build_chat_prompt(
        prompt: str, context: str | None = None, examples: list[Doc] | None = None
    ) -> list[BaseMessage]:
        out = []

        if context:
            out.append(SystemMessage(content=context))

        if examples:
            for example in examples:
                out.append(HumanMessage(content=example.title))
                out.append(AIMessage(content=example.contents))

        out.append(HumanMessage(content=prompt))

        return out

    def build_completion_prompt(
        self,
        prompt: str,
        context: str | None = None,
        examples: list[Doc] | None = None,
    ) -> str:
        out = ""

        if context:
            out = context + "\n\n"

        if examples:
            for example in examples:
                out += (
                    self.example_fmt.format(input=example.title, output=example.contents) + "\n\n"
                )

        out += prompt + "\n"

        return out

    def get_num_tokens(self, msg: str | list[BaseMessage]) -> int:
        """Get the number of tokens that would be used for this example, including all necessary
        formatting."""
        if isinstance(msg, list):
            return self.model.get_num_tokens_from_messages(msg)

        return self.model.get_num_tokens(msg)
