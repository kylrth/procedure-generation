import asyncio

import weaviate
import yaml

import retrieval
from dataset import Doc

from .interface import Response, System
from .model import Model


## PROMPTS ##

_prompt_query_gen_inst = (
    "Please output high-level steps to complete the task below.\n\n"
    "Then, given this high-level solution, think carefully step by step and provide search engine "
    "queries for knowledge that you need to refine the solution to the question.\n\n"
    "The output should be in YAML format, with the steps listed under `steps:` and the queries "
    "under `queries:`."
)
_prompt_query_gen = {
    "lcstep": "I want to create {query} using resources like {_input}. ",
    "recipenlg": "I want to make a {query} using {_input}. ",
    "champ": (
        "Given the following hints:\n\n"
        "{_input}\n\n"
        "I want to solve the following math problem:\n\n"
        "{query}\n\n"
    ),
}


class AAG(System):
    model: Model
    skills: weaviate.collections.Collection
    k: int
    dataset: str

    def __init__(self, model: Model, skills: weaviate.collections.Collection, k: int, dataset: str):
        """Create a new AAG system that maintains the skill library in the Weaviate instance.

        The model will only return one result when calling generate or agenerate.

        If the weaviate store already contains a collection named "Skills", that collection is
        assumed to contain the bootstrapped skills.
        """
        self.model = model
        self.skills = skills
        self.k = k
        self.dataset = dataset

    def generate(self, query: str, _input: str) -> Response:
        return asyncio.run(self.agenerate(query, _input))

    async def agenerate(self, query: str, _input: str) -> Response:
        queries = await self.queries_relevant_to(query, _input)

        docs: list[list[Doc]] = []
        for query in queries:
            docs.append(self.get_docs(query))

        for q in asyncio.as_completed([self.only_useful_docs(q, d) for q, d in zip(queries, docs)]):
            useful = await q

    async def queries_relevant_to(self, query: str, _input: str) -> list[str]:
        prompt = self.model.build_prompt(
            _prompt_query_gen[self.dataset].format(query=query, _input=_input),
            context=_prompt_query_gen_inst,
        )

        # Have the model generate an output, and check that it contains the query list. If it
        # doesn't, ask again.
        for _ in range(3):
            completion = (await self.model.agenerate(prompt))[0]
            out = yaml.safe_load(completion)

            if "queries" in out and isinstance(out["queries"], list):
                break
        else:
            raise ValueError("could not get good response from LLM after 3 tries")

        return out["queries"]

    def get_docs(self, query: str) -> list[Doc]:
        """Returns the skills relevant to the query as docs.

        TODO this will be list[Procedures] once we build the AAG-specific vector store
        """
        embedded_query = retrieval.get_embeds([query])[0]

        res = self.skills.query.near_vector(
            near_vector=embedded_query.tolist(),
            limit=self.k,
            return_properties=["title", "contents"],
        )

        out = []
        for obj in res.objects:
            out.append(Doc(obj.properties["title"], obj.properties["contents"]))

        return out

    async def only_useful_docs(self, query: str, docs: list[Doc]) -> list[Doc]:
        pass
