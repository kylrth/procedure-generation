"""Callback Handler that writes to a file."""

from typing import Any, TextIO

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils.input import print_text


class FileHandleCallbackHandler(BaseCallbackHandler):
    """Callback Handler that writes to an *already open* file.

    Opening and closing must be managed by callers.

    This replicates the functionality of Langchain's FileCallbackHandler.
    """

    def __init__(self, f: TextIO) -> None:
        """Initialize callback handler."""
        self.file = f

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **_: Any) -> None:
        """Print out that we are entering a chain."""
        _ = inputs

        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        print_text(f"\n\n> Entering new {class_name} chain...", end="\n", file=self.file)

    def on_chain_end(self, outputs: dict[str, Any], **_: Any) -> None:
        """Print out that we finished a chain."""
        _ = outputs

        print_text("\n> Finished chain.", end="\n", file=self.file)

    def on_agent_action(self, action: AgentAction, color: str | None = None, **kwargs: Any) -> Any:
        """Run on agent action."""
        _ = color
        _ = kwargs

        print_text(action.log, file=self.file)

    def on_tool_end(
        self,
        output: str,
        color: str | None = None,
        observation_prefix: str | None = None,
        llm_prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        _ = color
        _ = kwargs

        if observation_prefix is not None:
            print_text(f"\n{observation_prefix}", file=self.file)
        print_text(output, file=self.file)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}", file=self.file)

    def on_text(self, text: str, color: str | None = None, end: str = "", **kwargs: Any) -> None:
        """Run when agent ends."""
        _ = color
        _ = kwargs
        print_text(text, end=end, file=self.file)

    def on_agent_finish(self, finish: AgentFinish, color: str | None = None, **kwargs: Any) -> None:
        """Run on agent end."""
        _ = color
        _ = kwargs
        print_text(finish.log, end="\n", file=self.file)
