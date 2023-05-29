from typing import List, Optional, Tuple, Union

from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


class Model:
    """Model provides an abstraction around chat and text completion models suitable for this
    project, so that we can use chat and completion models with the same code."""

    # This is the string used to format examples into the prompts for completion models.
    example_fmt: str = "===BEGIN EXAMPLE===\n{title}\n{recipe}\n===END EXAMPLE==="

    def __init__(
        self, model: BaseLanguageModel, chat: bool = False, example_fmt: Optional[str] = None
    ):
        self.model = model
        self.chat = chat
        if example_fmt:
            self.example_fmt = example_fmt

    @classmethod
    def from_full_name(cls, full_name: str, example_fmt: Optional[str] = None, **kwargs) -> "Model":
        """Create a new model from a full name, which includes the service and model name.

        The full_name parameter combines the name of an LLM service (e.g. "openai") with the name of
        a model (e.g. "gpt-3.5-turbo"), separated by a hyphen (e.g. "openai-gpt-3.5-turbo").

        All kwargs are passed to the langchain model constructor.
        """
        service, model = full_name.lower().split("-", 1)

        factory, chat = cls.get_model_factory(service, model)

        model = factory(model=model, **kwargs)

        return cls(model, chat, example_fmt)

    @staticmethod
    def get_model_factory(service: str, model: str) -> Tuple[type, bool]:
        """Returns the factory for the specified model from langchain, and whether the model is a
        chat model."""
        if service == "openai":
            if "gpt-4" in model or "gpt-3.5-turbo" in model:
                return ChatOpenAI, True
            return OpenAI, False

        raise NotImplementedError(f"service '{service}' not added yet")

    def __call__(
        self,
        prompt: str,
        context: Optional[str] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
    ) -> List[str]:
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

            Human: {context}

                   {examples[0][0]}
            Model: {examples[0][1]}
            Human: {examples[1][0]}
            Model: {examples[1][1]}
            ...
            Human: {prompt}
        """
        prompt = self.build_prompt(prompt, context, examples)
        return self.generate(prompt)

    def generate(self, final_prompt: Union[str, List[BaseMessage]]) -> List[str]:
        """Calls the model on the already-constructed prompt. Can be used in conjunction with
        build_prompt to re-use constructed prompts."""
        out = self.model.generate([final_prompt])

        # In the future we may surface other information from the results, but for now we just need
        # the strings.
        return [r.text for r in out.generations[0]]

    def build_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
    ) -> Union[str, List[BaseMessage]]:
        """Builds the final prompt as described in documentation for __call__, but returns the
        constructed prompt instead of running the model.

        The type of the output depends on whether this is a chat or completion model."""
        if self.chat:
            return self.build_chat_prompt(prompt, context, examples)

        return self.build_completion_prompt(prompt, context, examples)

    @staticmethod
    def build_chat_prompt(
        prompt: str, context: Optional[str] = None, examples: Optional[List[Tuple[str, str]]] = None
    ) -> List[BaseMessage]:
        out = []

        if context:
            out.append(SystemMessage(content=context))

        if examples:
            for example in examples:
                out.append(HumanMessage(content=example[0]))
                out.append(AIMessage(content=example[1]))

        out.append(HumanMessage(content=prompt))

        return out

    def build_completion_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        out = ""

        if context:
            out = context + "\n\n"

        if examples:
            for example in examples:
                out += self.example_fmt.format(title=example[0], recipe=example[1]) + "\n\n"

        out += prompt + "\n"

        return out
