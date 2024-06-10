import re
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
        summaries: list[str] = []
        for q in queries:
            proc_list = await self.skills.search(q, self.k)
            procs.append(proc_list)
            summaries.append(await self.create_summary(logger, q, proc_list))

        logger.write(f"got {len(queries)} search queries from {self.model.name}:\n")
        for i, q in enumerate(queries):
            logger.write(f"- {q}\n")
            for p in procs[i]:
                logger.write(f"  - {p.output}\n")

        init_steps = await self.get_rag_response(logger, query, input_)
        candidate = Procedure(input_, query, init_steps)

        logger.write("BEGIN RAG CANDIDATE\n")
        logger.write(textwrap.indent(candidate.format_steps(), "  ") + "\n")
        logger.write("END RAG CANDIDATE\n")

        candidate.steps = await self.update_steps(logger, candidate, queries, summaries)

        return Response(
            answer=candidate.steps,
            model=self.model.name,
            # TODO token counts
        )

    async def get_rag_response(
        self, logger: log.InstanceLogger, query: str, input_: str
    ) -> list[str]:
        q = f"{query} using {input_}"
        procs = await self.skills.search(q, self.k)
        context = self.format_procedures(procs)
        msg_prompt = (
            context + "\n\n" + RAG._prompt_inst[self.dataset].format(query=query, input_=input_)
        )
        sys_prompt = RAG._instructions[self.dataset]
        sys_prompt += (
            " Think carefully about your steps and enclose any steps you are uncertain about in "
            "the format like '[[ <step> ]]'"
        )
        out = self.model.build_prompt(msg_prompt, sys_prompt)

        logger.write(f"prompt to model {self.model.name}:\n")
        logger.log_prompt(out)
        completion = self.parse_completion(await self.model.generate(out))
        return completion

    _instructions: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please update the provided high-level steps to accomplish the specified goal using "
            "the LangChain Python library. Focus more on improving the uncertain steps enclosed in "
            "'[[]]'. Don't include code, extraneous commentary, or examples, but do refer "
            "to the specific LangChain APIs (or other APIs) used in each step. Don't produce any "
            "text other than the list of steps. Use any of the provided reference answers to "
            "relevant questions on the steps to achieve the specified goal."
        ),
        "recipenlg": (
            "Please update the provided high-level steps to create a recipe for the specified "
            "food. Focus more on improving the uncertain steps enclosed in '[[]]'. Don't "
            "include extraneous commentary, or examples, but do refer to the special "
            "characteristics and state of the ingredients used in each step. Don't produce any "
            "text other than the list of steps. Use any of the provided reference answers to "
            "relevant questions on the steps to achieve the specified goal."
        ),
        "champ": (
            "Please update the provided high-level steps to solve the given math problem. Focus "
            "more on improving the uncertain steps enclosed in '[[]]'. Don't include code, "
            "extraneous commentary, or examples, but do refer to the concepts and hints used in "
            "each step. Don't produce any text other than the list of steps. Use any of the "
            "provided reference answers to relevant questions on the steps to achieve the "
            "specified goal."
        ),
    }

    _prompt_inst: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please update the list of steps to accomplish '{query}' using the knowledge "
            "above. Create and use these resources in your response: {input_}. "
            "Please output only the updated steps. Your response should start with '1.'. "
            "The final response should not contain direct references to the knowledge above."
        ),
        "recipenlg": (
            "Please update the list of steps to accomplish '{query}' using the knowledge "
            "above. Use these ingredients in your response: {input_}. "
            "Please output only the updated steps. Your response should start with '1.'. "
            "The final response should not contain direct references to the knowledge above."
        ),
        "champ": (
            "Please update the list of steps to solve '{query}' using the knowledge above. "
            "Use this additional information in preparing your response: {input_}. "
            "Please output only the updated steps. Your response should start with '1.'. "
            "The final response should not contain direct references to the knowledge above."
        ),
    }

    def format_summaries(self, queries: list[str], summaries: list[str]) -> str:
        sum_str = ""
        for q, summary in zip(queries, summaries):
            sum_str += f"Q: {q}\nA: {summary}\n\n"

        return sum_str.rstrip()

    async def update_steps(
        self,
        logger: log.InstanceLogger,
        candidate: Procedure,
        queries: list[str],
        summaries: list[str],
    ) -> list[str]:
        summary_str = self.format_summaries(queries, summaries)
        msg_prompt = (
            f"[BEGIN KNOWLEDGE]\n{summary_str}\n[END KNOWLEDGE]"
            "\n\n"
            f"[BEGIN STEPS]\n{candidate.format_steps()}\n[END STEPS]"
            "\n\n"
            + self._prompt_inst[self.dataset].format(
                query=candidate.output, input_=candidate.input_
            )
        )

        prompt = self.model.build_prompt(
            prompt=msg_prompt, context=self._instructions[self.dataset]
        )
        logger.write("Prompt to update RAG response:\n")
        logger.log_prompt(prompt)
        completion = await self.model.generate(prompt)
        return self.parse_completion(completion)

    _prompt_query_gen_inst: ClassVar[str] = (
        "Please output high-level steps to complete the task below.\n\n"
        "Then, given this high-level solution, think carefully step by step and provide 4 search "
        "engine queries for knowledge that you need to refine the solution to the question.\n\n"
        "The output should be 'steps:' followed by a bulleted list with elements starting with "
        "'- ', and then 'queries:' followed by another bulleted list."
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
    _query_gen_out_prefix: ClassVar[re.Pattern] = re.compile(
        r"(\*\*)?queries:(\*\*)?", flags=re.IGNORECASE
    )

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
            completion = await self.model.generate(prompt)
            logger.write(f"asked {self.model.name} for relevant queries; model said:\n")
            logger.write(textwrap.indent(completion, "  ") + "\n")

            try:
                # find the first match
                match = next(iter(self._query_gen_out_prefix.finditer(completion)))

                lines = completion[match.end() :].strip().split("\n")
            except StopIteration:
                logger.write("ERROR: 'queries:' not found in the above completion; trying again\n")
                continue

            for line in lines:
                if not line.startswith("- "):
                    logger.write("ERROR: the query list was badly formatted\n")
                    continue

            return [line[2:] for line in lines]

        raise ValueError("could not get good response from LLM after 3 tries")

    def format_procedures(self, procs: list[Procedure]) -> str:
        out = ""
        for p in procs:
            out += f"\n\n{RAG._example_name[self.dataset]} '{p.output}' using {p.input_}:\n\n"
            out += p.format_steps()

        return out[2:]  # skip first "\n\n"

    async def create_summary(
        self, logger: log.InstanceLogger, query: str, procs: list[Procedure]
    ) -> str:
        context = self.format_procedures(procs)
        sys_instruction = (
            "[Instruction]\n"
            "You are a human expert whose job is to summarise the retrieved "
            "information below to answer the question. Please include the "
            "information only from the provided knowledge and make sure "
            "that the summary is complete, short and concise. Avoid introductory and "
            "closing lines at the start and end of your response. "
            "Don't directly refer to the titles in the provided knowledge when generating the "
            "summary."
        )
        msg_prompt = (
            f"[BEGIN QUESTION]\n{query}\n[END QUESTION]\n\n"
            f"[BEGIN INFORMATION]\n{context}\n[END INFORMATION]"
        )
        prompt = self.model.build_prompt(
            prompt=msg_prompt,  # Human Message
            context=sys_instruction,  # System Message
        )
        logger.log_prompt(prompt)

        completion = await self.model.generate(prompt)

        return completion

    _create_candidate_instructions: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please write high-level steps to use LangChain to {query} "
            "using these resources: {input_}. Refer to the similar procedures below for any "
            "useful information. Your response should begin with '1.'."
        ),
        "recipenlg": (
            "Please create recipe instructions for making {query} using these "
            "ingredients: {input_}. Refer to similar recipes below for any useful information. "
            "Your response should begin with '1.'."
        ),
        "champ": (
            "Given the following hints:\n\n"
            "{input_}\n\n"
            "I want to solve the following math problem:\n\n"
            "{query}\n\n"
            "Please create a candidate step-by-step solution by referring to useful information "
            "from the similar solutions below. Your response should begin with '1.'."
        ),
    }

    async def create_candidate(
        self, logger: log.InstanceLogger, query: str, input_: str, procs: list[Procedure]
    ) -> list[str]:
        context = self.format_procedures(procs)
        msg_prompt = (
            context + "\n\n" + RAG._prompt_inst[self.dataset].format(query=query, input_=input_)
        )
        prompt = self.model.build_prompt(
            prompt=msg_prompt,  # self.format_procedures(procs),
            context=RAG._instructions[self.dataset],
        )
        logger.log_prompt(prompt)

        completion = await self.model.generate(prompt)

        return self.parse_completion(completion)

    _update_candidate_instructions: ClassVar[dict[str, str]] = {
        "lcstep": (
            "We have a draft procedure to {query} using these resources: {input_}, "
            "but it may not yet have all the information needed to complete the task."
        ),
        "recipenlg": (
            "We have a draft recipe to make {query} using these ingredients: {input_}.\n\n"
            "But it may not yet be completely correct."
        ),
        "champ": (
            "Given the following hints:\n\n"
            "{input_}\n\n"
            "I want to solve the following math problem:\n\n"
            "{query}\n\n"
            "We have a draft solution, but it may not be complete or correct yet."
        ),
    }

    async def update_candidate(
        self, logger: log.InstanceLogger, candidate: Procedure, procs: list[Procedure]
    ) -> list[str]:
        prompt = self.model.build_prompt(
            prompt=f"BEGIN DRAFT\n{candidate.format_steps()}\nEND DRAFT\n\n"
            + self.format_procedures(procs)
            + "\n\nBased on the additional information above, update the draft if required. "
            "Please output only the updated draft. Your response should start with '1.'.",
            context=self._update_candidate_instructions[self.dataset].format(
                query=candidate.output,
                input_=candidate.input_,
            ),
        )
        logger.log_prompt(prompt)

        completion = await self.model.generate(prompt)

        return self.parse_completion(completion)
