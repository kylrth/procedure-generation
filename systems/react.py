from typing import ClassVar, cast

from langchain import hub
from langchain.agents import AgentExecutor, BaseMultiActionAgent, create_react_agent
from langchain.tools import BaseTool, tool
from langchain_core.callbacks import CallbackManager
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableConfig

import retrieval
from dataset import LinearProcedure, create_graphs_for_graph_store
from model import Model
from utils import log
from utils.lc import FileHandleCallbackHandler

from .interface import Response, System


class ReAct(System):
    """Baseline ReAct system based on the original ReAct work: https://react-lm.github.io/.

    Based on the Langchain implementation referenced here:
    https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react.html.
    """

    _dataset_details: ClassVar[dict[str, dict[str, str]]] = {
        "lcstep": {
            "system_msg": (
                "Your goal is to generate high-level steps to accomplish a specific task using the "
                "LangChain Python library. Don't include code, extraneous commentary, or examples, "
                "but do refer to the specific LangChain APIs (or other APIs) used in each step."
            ),
            "name": "procedure",
            "example_q": "split a long text into chunks based on token count using tiktoken",
            "example_t": "I need to find examples splitting long text into chunks",
            "example_i": "split a long text into chunks by token count",
            "example_o": "<example procedures which may do text splitting>",
            "example_a": (
                "1. Initialize a `langchain.text_splitter.TokenTextSplitter` with the desired "
                "chunk size and overlap.\n"
                "2. Split the text into chunks by calling the `split_text` method on the "
                "`TokenTextSplitter` object with the text as the parameter."
            ),
        },
        "recipenlg": {
            "system_msg": (
                "Your goal is to generate high-level steps to create a recipe for the specified "
                "food. Don't include extraneous commentary, or examples, but do refer to the "
                "special characteristics and state of the ingredients used in each step."
            ),
            "name": "recipe",
            "example_q": "Mandarin Orange Cake",
            "example_t": "I need to find recipes that bake with mandarin oranges",
            "example_i": "baking with mandarin oranges",
            "example_o": "<relevant recipes>",
            "example_a": (
                "1. Beat together and put in a 9 x 13-inch pan.\n"
                "2. Bake 25 to 30 minutes in a 350° oven.\n"
                "3. Cool, then place topping on top."
            ),
        },
        "champ": {
            "system_msg": (
                "Your goal is to generate high-level steps to solve the given math problem. Don't "
                "include code, extraneous commentary, or examples, but do refer to the concepts "
                "and hints used in each step. The final step should start with 'The answer is '."
            ),
            "name": "example",
            "example_q": (
                "Let f(x) be a polynomial of degree n with integer coefficients. If there are "
                "three different integers a, b, c, such that f(a)=f(b)=f(c)=-1, then at most how "
                "many integer-valued roots can this polynomial have?"
            ),
            "example_t": "I need to find math problems dealing with integer-valued roots",
            "example_i": "integer-valued roots of a polynomial",
            "example_o": "<relevant problems and solutions>",
            "example_a": (
                "1. Let g(x)=f(x)+1, so a, b, c are the three roots of g(x).\n"
                "2. Thus, g(x)=(x-a)(x-b)(x-c)h(x) for some polynomial h(x).\n"
                "...\n"
                "6. Thus, we have a contradiction, and no such k exists.\n"
                "7. The answer is At most 0 integer-valued roots (i.e., no integer roots)"
            ),
        },
    }

    _template: ClassVar[str] = (
        "{system_msg} You have access to the following tools:\n"
        "\n"
        "{{tools}}\n"
        "\n"
        "Use the following format:\n"
        "\n"
        "Question: {example_q}\n"
        "Thought: {example_t}\n"
        "Action: search\n"
        "Action Input: {example_i}\n"
        "Observation: {example_o}\n"
        "... (this pattern of Thought->Action->Action Input->Observation can repeat N times)\n"
        "Thought: I think I know enough to produce the final answer\n"
        "Final Answer:\n"
        "{example_a}\n"
        "\n"
        "The answer to this question will not appear exactly in the database; you need to write an "
        "answer using the knowledge you gain by searching for similar {name}s. Feel free to "
        "search several times with different queries until you have all the information you need "
        "to answer the question.\n"
        "\n"
        "Begin!\n"
        "\n"
        "Question: {{input}}\n"
        "Thought:{{agent_scratchpad}}\n"
    )

    def __init__(
        self, model: Model, dataset: str, graphs: retrieval.GraphProcedureStore, k: int, hs: bool
    ):
        self.dataset = dataset
        self.graphs = graphs
        self.k = k
        self.model = model
        self.hs = hs

        prompt = cast(PromptTemplate, hub.pull("hwchase17/react"))
        prompt.template = self._template.format(**self._dataset_details[dataset])
        self.agent = AgentExecutor(
            agent=cast(
                BaseMultiActionAgent, create_react_agent(model.model, [self.search_tool], prompt)
            ),
            tools=[self.search_tool],
            callbacks=CallbackManager(handlers=[]),  # handlers will be managed in `generate`
            handle_parsing_errors=True,
        )

    _search_tool: BaseTool | None = None

    _example_name: ClassVar[dict[str, str]] = {
        "lcstep": "procedures",
        "recipenlg": "recipes",
        "champ": "examples of math problems and solutions",
    }

    @property
    def search_tool(self) -> BaseTool:
        if self._search_tool is None:

            async def search(query: str) -> str:
                if self.hs:
                    procedures = await self.graphs.hierarchical_retrieval(
                        query, k=2 * self.k, k2=self.k
                    )
                else:
                    procedures = await self.graphs.search(query, self.k)

                return "\n\n".join(str(p) for p in procedures)

            search.__doc__ = (
                f"Search a database for known {self._example_name[self.dataset]} related to the "
                "query."
            )

            self._search_tool = cast(BaseTool, tool(search))

        return self._search_tool

    async def generate(self, logger: log.InstanceLogger, query: str, input_: str) -> Response:
        res = await self.agent.ainvoke(
            {"input": f"{query} using {input_}"},
            # use this instance's log file
            config=RunnableConfig(callbacks=[FileHandleCallbackHandler(logger._wrapped)]),
        )

        proc = LinearProcedure(input_, query, self.parse_completion(res["output"]))
        graph = await create_graphs_for_graph_store(
            logger, -1, proc, self.model, self.dataset, save_pkl=False
        )
        return Response(answer=graph, model="")
