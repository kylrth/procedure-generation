import textwrap
from typing import Any, ClassVar

import yaml

from dataset import Procedure
from retrieval import ProcedureStore
from utils import log

from .interface import Response, System
from .model import Model
from .rag import RAG


def limited_yaml(obj: dict[str, list[Any]]) -> dict[str, list[str]]:
    """Update an object parsed from YAML so that we know all the list elements are strings.

    Models often make mistakes with the edge cases of YAML, but asking for YAML output makes
    parsing easier. So we'll parse the YAML but then reconvert all list elements back to strings.
    """
    for list_value in obj.values():
        for i in range(len(list_value)):
            if not isinstance(list_value[i], str):
                list_value[i] = yaml.dump(list_value[i])

    return obj


class AAG(System):
    model: Model
    skills: ProcedureStore
    k: int
    dataset: str

    def __init__(self, model: Model, skills: ProcedureStore, k: int, dataset: str):
        """Create a new AAG system that maintains the skill library in the Weaviate instance.

        The model will only return one result when calling generate or agenerate.

        If the weaviate store already contains a collection named "Skills", that collection is
        assumed to contain the bootstrapped skills.
        """
        self.model = model
        self.skills = skills
        self.k = k
        self.dataset = dataset

    async def generate(self, logger: log.InstanceLogger, query: str, input_: str) -> Response:
        queries = await self.queries_relevant_to(logger, query, input_)

        procs: list[list[Procedure]] = []
        for query in queries:
            procs.append(await self.skills.search(query, self.k))

        logger.write(f"got {len(queries)} search queries from {self.model.name}:\n")
        for i, q in enumerate(queries):
            logger.write(f"- {q}\n")
            for p in procs[i]:
                logger.write(f"  - {p.output}\n")

        # TODO in the future try filtering

        # iteratively prompt LLM with the first procedure from each query, then the second ones,
        # etc.
        candidate: Procedure = Procedure(input_, query, [])
        for i, proc_set in enumerate(zip(*procs, strict=True)):
            if len(candidate.steps) == 0:
                candidate.steps = await self.create_candidate(query, input_, proc_set)
            else:
                candidate.steps = await self.update_candidate(candidate, proc_set)

            logger.write(
                f"candidate after looking at the {i + 1}th closest procedures for all "
                f"{len(proc_set)} queries:\n"
            )
            logger.write(textwrap.indent(candidate.format_steps(), "  ") + "\n")

        return Response(
            answer=candidate.steps,
            model=self.model.name,
            # TODO token counts
        )

    _prompt_query_gen_inst: ClassVar[str] = (
        "Please output high-level steps to complete the task below.\n\n"
        "Then, given this high-level solution, think carefully step by step and provide 3-5 search "
        "engine queries for knowledge that you need to refine the solution to the question.\n\n"
        "The output should be in YAML format, with the steps listed under `steps:` and the queries "
        "under `queries:`."
    )
    _prompt_query_gen: ClassVar[dict[str, str]] = {
        "lcstep": "I want to create {query} using these resources: {input_}. ",
        "recipenlg": "I want to make a {query} using these ingredients: {input_}. ",
        "champ": (
            "Given the following hints:\n\n"
            "{input_}\n\n"
            "I want to solve the following math problem:\n\n"
            "{query}\n\n"
        ),
    }

    async def queries_relevant_to(
        self, logger: log.InstanceLogger, query: str, input_: str
    ) -> list[str]:
        prompt = self.model.build_prompt(
            self._prompt_query_gen[self.dataset].format(query=query, input_=input_),
            context=self._prompt_query_gen_inst,
        )

        # Have the model generate an output, and check that it contains the query list. If it
        # doesn't, ask again.
        for _ in range(3):
            completion = (await self.model.agenerate(prompt))[0]
            logger.write(f"asked {self.model.name} for relevant queries; model said:\n")
            logger.write(textwrap.indent(completion, "  ") + "\n")

            out = yaml.safe_load(completion)

            if "queries" not in out or not isinstance(out["queries"], list):
                logger.write(f"queries were wrong: {out}\n")
                continue
            try:
                out = limited_yaml(out)
            except (KeyError, TypeError) as e:
                logger.write(f"bad YAML: {e!s}\n")
                continue

            break
        else:
            raise ValueError("could not get good response from LLM after 3 tries")

        return out["queries"]

    def build_context(self, procs: list[Procedure]) -> str:
        out = ""
        for p in procs:
            out += f"\n\n{RAG._example_name[self.dataset]} '{p.output}' using {p.input_}:\n\n"
            out += p.format_steps()

        return out[2:]  # skip first "\n\n"

    _prompt_candidate: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please create a candidate step-by-step solution to use LangChain to create {query} "
            "using these resources: {input_}. Borrow useful information from the following similar "
            "solutions:\n\n"
            "{procedures}\n\n"
            "Your response should begin with '1.'."
        ),
        "recipenlg": (
            "Please create a candidate step-by-step recipe for making {query} using these "
            "ingredients: {input_}. Borrow useful information from the following similar "
            "recipes:\n\n"
            "{procedures}\n\n"
            "Your response should begin with '1.'."
        ),
        "champ": (
            "Given the following hints:\n\n"
            "{input_}\n\n"
            "I want to solve the following math problem:\n\n"
            "{query}\n\n"
            "Please create a candidate step-by-step solution by borrowing useful information from "
            "the following similar solutions:\n\n"
            "{procedures}\n\n"
            "Your response should begin with '1.'."
        ),
    }

    async def create_candidate(self, query: str, input_: str, procs: list[Procedure]) -> list[str]:
        prompt = self.model.build_prompt(
            self._prompt_candidate[self.dataset].format(
                query=query, input_=input_, procedures=self.build_context(procs)
            )
        )

        completion = (await self.model.agenerate(prompt))[0]

        return self.parse_completion(completion)

    _prompt_update_candidate: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please update this candidate step-by-step solution to use LangChain to create {query} "
            "using these resources: {input_}:\n\n"
            "[BEGIN CANDIDATE]\n"
            "{candidate}\n"
            "[END CANDIDATE]\n\n"
            "Borrow useful information from the following similar solutions:\n\n"
            "{procedures}\n\n"
            "Your response should begin with '1.'."
        ),
        "recipenlg": (
            "Please update this candidate step-by-step recipe for making {query} using these "
            "ingredients: {input_}:\n\n"
            "[BEGIN CANDIDATE]\n"
            "{candidate}\n"
            "[END CANDIDATE]\n\n"
            "Borrow useful information from the following similar recipes:\n\n"
            "{procedures}\n\n"
            "Your response should begin with '1.'."
        ),
        "champ": (
            "Given the following hints:\n\n"
            "{input_}\n\n"
            "I want to solve the following math problem:\n\n"
            "{query}\n\n"
            "Please update this candidate step-by-step solution to do so:\n\n"
            "[BEGIN CANDIDATE]\n"
            "{candidate}\n"
            "[END CANDIDATE]\n\n"
            "Borrow useful information from the following similar solutions:\n\n"
            "{procedures}\n\n"
            "Your response should begin with '1.'."
        ),
    }

    async def update_candidate(self, candidate: Procedure, procs: list[Procedure]) -> list[str]:
        prompt = self.model.build_prompt(
            self._prompt_update_candidate[self.dataset].format(
                query=candidate.output,
                input_=candidate.input_,
                candidate=candidate.format_steps(),
                procedures=self.build_context(procs),
            )
        )

        completion = (await self.model.agenerate(prompt))[0]

        return self.parse_completion(completion)
