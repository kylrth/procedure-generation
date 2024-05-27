# ruff: noqa: E501

import unittest

from dataset.LCStep import format_procedure


class TestFormatOutput(unittest.TestCase):
    def test_all(self):
        cases = {
            "empty": (
                "",
                "",
            ),
            "no_delims": (
                "This is some text.\n",
                "This is some text.\n",
            ),
            "no_newline": (
                "no newline",
                "no newline\n",
            ),
            "with_delims": (
                """BEGIN EXAMPLE
stream the final output of an agent using FinalStreamingStdOutCallbackHandler
1. Initialize your language model with `langchain.llms.OpenAI`, setting `streaming=True`, `temperature=0`, and `callbacks` as a list containing a new instance of `langchain.callbacks.streaming_stdout_final_only.FinalStreamingStdOutCallbackHandler`.
2. Load your tools using `langchain.agents.load_tools` with the names of the tools and the language model as parameters.
3. Initialize your agent using `langchain.agents.initialize_agent` with the tools, language model, and agent type as parameters.
4. Run your agent using the `run` method with the request as a string.
Side note: To use a custom answer prefix, initialize the language model as above but with `answer_prefix_tokens` parameter in `FinalStreamingStdOutCallbackHandler` set to a list containing your custom answer prefix. The callback automatically strips whitespaces and new line characters when comparing to `answer_prefix_tokens`. If you don't know the tokenized version of your answer prefix, you can determine it by creating a custom callback handler that inherits from `langchain.callbacks.base.BaseCallbackHandler` and overrides the `on_llm_new_token` method to print every token. To stream the answer prefix itself, set `stream_prefix = True` in `FinalStreamingStdOutCallbackHandler`.
END EXAMPLE""",
                """stream the final output of an agent using FinalStreamingStdOutCallbackHandler
1. Initialize your language model with `langchain.llms.OpenAI`, setting `streaming=True`, `temperature=0`, and `callbacks` as a list containing a new instance of `langchain.callbacks.streaming_stdout_final_only.FinalStreamingStdOutCallbackHandler`.
2. Load your tools using `langchain.agents.load_tools` with the names of the tools and the language model as parameters.
3. Initialize your agent using `langchain.agents.initialize_agent` with the tools, language model, and agent type as parameters.
4. Run your agent using the `run` method with the request as a string.
Side note: To use a custom answer prefix, initialize the language model as above but with `answer_prefix_tokens` parameter in `FinalStreamingStdOutCallbackHandler` set to a list containing your custom answer prefix. The callback automatically strips whitespaces and new line characters when comparing to `answer_prefix_tokens`. If you don't know the tokenized version of your answer prefix, you can determine it by creating a custom callback handler that inherits from `langchain.callbacks.base.BaseCallbackHandler` and overrides the `on_llm_new_token` method to print every token. To stream the answer prefix itself, set `stream_prefix = True` in `FinalStreamingStdOutCallbackHandler`.
""",
            ),
        }

        for name in cases:
            with self.subTest(name):
                got = format_procedure.format_output(cases[name][0])
                want = cases[name][1]
                self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
