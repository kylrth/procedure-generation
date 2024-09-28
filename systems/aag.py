import random
import re
import sys
import textwrap
from typing import ClassVar

from dataset import LinearProcedure
from retrieval import ProcedureStore
from utils import log

from .interface import Response, System
from .model import Model
from .rag import RAG


class AAG(System):
    model: Model
    skills: ProcedureStore
    k: int
    dataset: str
    summarize: bool
    use_critic: bool

    def __init__(
        self,
        model: Model,
        skills: ProcedureStore,
        k: int,
        dataset: str,
        summarize: bool,
        critic: bool,
        n_queries: int,
    ):
        """Create a new AAG system that maintains the skill library in the Weaviate instance.

        The model will only return one result when calling generate or agenerate.

        If the weaviate store already contains a collection named "Skills", that collection is
        assumed to contain the bootstrapped skills.
        """
        self.model = model
        self.skills = skills
        self.k = k
        self.dataset = dataset
        self.summarize = summarize
        self.use_critic = critic
        self.n_queries = n_queries

    async def generate(self, logger: log.InstanceLogger, query: str, input_: str) -> Response:
        queries = await self.queries_relevant_to(logger, query, input_)

        procs: list[list[LinearProcedure]] = []
        summaries: list[str] = []
        for q in queries:
            proc_list = await self.skills.search(q, self.k)
            procs.append(proc_list)
            if self.summarize:
                summaries.append(await self.create_summary(logger, q, proc_list))

        logger.write(f"got {len(queries)} search queries from {self.model.name}:\n")
        for i, q in enumerate(queries):
            logger.write(f"- {q}\n")
            for p in procs[i]:
                logger.write(f"  - {p.output}\n")

        init_steps = await self.get_rag_response(logger, query, input_)
        candidate = LinearProcedure(input_, query, init_steps)

        logger.write("BEGIN RAG CANDIDATE\n")
        logger.write(textwrap.indent(candidate.format_steps(), "  ") + "\n")
        logger.write("END RAG CANDIDATE\n")

        if self.summarize:
            knowledge_str = self.format_summaries(queries, summaries)
        else:
            flat_proc_list = [p for p_list in procs for p in p_list]
            knowledge_str = self.format_procedures(flat_proc_list)

        candidate.steps = await self.update_steps(logger, candidate, knowledge_str)

        if self.use_critic:
            candidate.steps = await self.check_with_validator_and_modify(
                logger, candidate, knowledge_str, max_updates=3
            )

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

    _no_summ_instructions: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please update the provided high-level steps to accomplish the specified goal using "
            "the LangChain Python library. Focus more on improving the uncertain steps enclosed in "
            "'[[]]'. Don't include code, extraneous commentary, or examples, but do refer "
            "to the specific LangChain APIs (or other APIs) used in each step. Don't produce any "
            "text other than the list of steps. Use any of the provided reference documentation to "
            "achieve the specified goal."
        ),
        "recipenlg": (
            "Please update the provided high-level steps to create a recipe for the specified "
            "food. Focus more on improving the uncertain steps enclosed in '[[]]'. Don't "
            "include extraneous commentary, or examples, but do refer to the special "
            "characteristics and state of the ingredients used in each step. Don't produce any "
            "text other than the list of steps. Use any of the provided reference recipes to "
            "achieve the specified goal."
        ),
        "champ": (
            "Please update the provided high-level steps to solve the given math problem. Focus "
            "more on improving the uncertain steps enclosed in '[[]]'. Don't include code, "
            "extraneous commentary, or examples, but do refer to the concepts and hints used in "
            "each step. Don't produce any text other than the list of steps. Use any of the "
            "provided reference problems and their solutions to achieve the "
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
        self, logger: log.InstanceLogger, candidate: LinearProcedure, knowledge_str: str
    ) -> list[str]:
        msg_prompt = (
            f"[BEGIN KNOWLEDGE]\n{knowledge_str}\n[END KNOWLEDGE]"
            "\n\n"
            f"[BEGIN STEPS]\n{candidate.format_steps()}\n[END STEPS]"
            "\n\n"
            + self._prompt_inst[self.dataset].format(
                query=candidate.output, input_=candidate.input_
            )
        )
        if self.summarize:
            sys_instruction = self._instructions[self.dataset]
        else:
            sys_instruction = self._no_summ_instructions[self.dataset]
        prompt = self.model.build_prompt(prompt=msg_prompt, context=sys_instruction)
        logger.write("Prompt to update RAG response:\n")
        logger.log_prompt(prompt)
        completion = await self.model.generate(prompt)
        return self.parse_completion(completion)

    _prompt_query_gen_inst: ClassVar[str] = (
        "Please output high-level steps to complete the task below.\n\n"
        "Then, given this high-level solution, think carefully step by step and provide "
        "{n_queries} search engine queries for knowledge that you need to refine the solution to "
        "the question.\n\n"
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
            context=self._prompt_query_gen_inst.format(n_queries=self.n_queries),
        )

        # Have the model generate an output, and check that it contains the query list. If it
        # doesn't, ask again. To ensure different responses even when using a caching proxy, we set
        # a different seed each time.
        rng = random.Random(28)
        for i in range(3):
            if i == 0:
                completion = await self.model.generate(prompt)
            else:
                completion = await self.model.generate(prompt, seed=rng.randint(0, sys.maxsize))
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

    def format_procedures(self, procs: list[LinearProcedure]) -> str:
        seen = set()
        out = ""
        for p in procs:
            if p in seen:
                continue
            out += f"\n\n{RAG._example_name[self.dataset]} '{p.output}' using {p.input_}:\n\n"
            out += p.format_steps()
            seen.add(p)

        return out[2:]  # skip first "\n\n"

    async def create_summary(
        self, logger: log.InstanceLogger, query: str, procs: list[LinearProcedure]
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

    """
    Validator checks for:
    1) If all inputs are used or not: Accept extra
    ingredients in the serving part but not while
    making components of the dish
    2) Completes the user goal or not: any change in flow of steps or
    adding some details.

    Suggest edits as a bulleted list. If no update required,
    respond 'NO UPDATE REQUIRED'
    """

    _validator_opt_inst: ClassVar[dict[str, str]] = {
        "lcstep": "",
        "recipenlg": (
            "For the provided recipe, do not penalize additional ingredients "
            "used for better serving or decorating and the utensils. However, "
            "there should not be extra ingredients used in making the components "
            "of the food."
        ),
        "champ": "",
    }

    async def validate_update(self, logger: log.InstanceLogger, candidate: LinearProcedure) -> str:
        sys_instruction = (
            "[INSTRUCTION]\nYou are a human critic whose job is to validate the "
            "provided procedure, propose the changes to be made and evaluate if "
            "the steps lead to the mentioned "
            "user goal or not. You should also assess if the quality of the steps "
            "can be improved by modifying the flow of the steps or adding "
            "more details to make it more clear and doable.\n\n"
            "Furthermore, it is very important for the procedure to use all the "
            "mentioned input resources. Carefully judge if the procedure uses "
            "all the resources and point out in your response if it misses "
            "something. {opt_inst}\n\n"
            "Please output only your edits in a bulleted list or if there "
            "are absolutely no edits, please strictly output only 'NO UPDATE REQUIRED'. "
            "You are required to strictly follow the mentioned output format."
        )
        msg_prompt = (
            f"[USER GOAL]\n{candidate.output}\n\n"
            f"[INPUT RESOURCES]\n{candidate.input_}\n\n"
            f"[BEGIN PROCEDURE]\n{candidate.format_steps()}\n[END PROCEDURE]"
        )
        prompt = self.model.build_prompt(
            prompt=msg_prompt,  # Human Message
            context=sys_instruction.format(
                opt_inst=self._validator_opt_inst[self.dataset]
            ),  # System Message
        )
        completion = await self.model.generate(prompt)

        logger.write("VALIDATOR PROMPT\n")
        logger.log_prompt(prompt)
        logger.write("BEGIN VALIDATOR ANSWER\n")
        logger.write(textwrap.indent(completion, "  ") + "\n")
        logger.write("END VALIDATOR ANSWER\n")
        return completion

    _perform_edits_instructions: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please update the provided high-level steps in accordance with the suggested edits "
            "to accomplish the specified goal using "
            "the LangChain Python library. Make sure all the other details remain unaltered. "
            "Don't include code, extraneous commentary, or examples, but do refer "
            "to the specific LangChain APIs (or other APIs) used in each step. Don't produce any "
            "text other than the list of steps. Use any of the provided reference answers to "
            "relevant questions on the steps to achieve the specified goal."
        ),
        "recipenlg": (
            "Please update the provided high-level steps in accordance with the suggested edits "
            "to create a recipe for the specified "
            "food. Make sure all the other details remain unaltered. Don't "
            "include extraneous commentary, or examples, but do refer to the special "
            "characteristics and state of the ingredients used in each step. Don't produce any "
            "text other than the list of steps. Use any of the provided reference answers to "
            "relevant questions on the steps to achieve the specified goal."
        ),
        "champ": (
            "Please update the provided high-level steps in accordance with the suggested edits "
            "to solve the given math problem. Make sure all the other details remain unaltered. "
            "Don't include code, extraneous commentary, or examples, but do refer to the "
            "concepts and hints used in "
            "each step. Don't produce any text other than the list of steps. Use any of the "
            "provided reference answers to relevant questions on the steps to achieve the "
            "specified goal."
        ),
    }

    _no_summ_perform_edits_instructions: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please update the provided high-level steps in accordance with the suggested edits "
            "to accomplish the specified goal using "
            "the LangChain Python library. Make sure all the other details remain unaltered. "
            "Don't include code, extraneous commentary, or examples, but do refer "
            "to the specific LangChain APIs (or other APIs) used in each step. Don't produce any "
            "text other than the list of steps. Use any of the provided reference documentation to "
            "achieve the specified goal."
        ),
        "recipenlg": (
            "Please update the provided high-level steps in accordance with the suggested edits "
            "to create a recipe for the specified "
            "food. Make sure all the other details remain unaltered. Don't "
            "include extraneous commentary, or examples, but do refer to the special "
            "characteristics and state of the ingredients used in each step. Don't produce any "
            "text other than the list of steps. Use any of the provided reference recipes to "
            "achieve the specified goal."
        ),
        "champ": (
            "Please update the provided high-level steps in accordance with the suggested edits "
            "to solve the given math problem. Make sure all the other details remain unaltered. "
            "Don't include code, extraneous commentary, or examples, but do refer to the "
            "concepts and hints used in "
            "each step. Don't produce any text other than the list of steps. Use any of the "
            "provided reference problems and their solutions to achieve the "
            "specified goal."
        ),
    }

    _perf_edit_inst: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please perform all the edits and update the list of steps to "
            "accomplish '{query}' using the knowledge "
            "above. Create and use these resources in your response: {input_}. "
            "Please output only the updated steps. Your response should start with '1.'. "
            "The final response should not contain direct references to the knowledge above."
        ),
        "recipenlg": (
            "Please perform all the edits and update the list of steps to "
            "accomplish '{query}' using the knowledge "
            "above. Use these ingredients in your response: {input_}. "
            "Please output only the updated steps. Your response should start with '1.'. "
            "The final response should not contain direct references to the knowledge above."
        ),
        "champ": (
            "Please perform all the edits and update the list of steps to "
            "solve '{query}' using the knowledge above. "
            "Use this additional information in preparing your response: {input_}. "
            "Please output only the updated steps. Your response should start with '1.'. "
            "The final response should not contain direct references to the knowledge above."
        ),
    }

    async def perform_validator_edits(
        self, logger: log.InstanceLogger, candidate: LinearProcedure, knowledge_str: str, edits: str
    ) -> list[str]:
        msg_prompt = (
            f"[BEGIN KNOWLEDGE]\n{knowledge_str}\n[END KNOWLEDGE]"
            "\n\n"
            f"[BEGIN STEPS]\n{candidate.format_steps()}\n[END STEPS]"
            "\n\n"
            f"[BEGIN EDITS]\n{edits}\n[END EDITS]"
            "\n\n"
            + self._perf_edit_inst[self.dataset].format(
                query=candidate.output, input_=candidate.input_
            )
        )

        if self.summarize:
            sys_instruction = self._perform_edits_instructions[self.dataset]
        else:
            sys_instruction = self._no_summ_perform_edits_instructions[self.dataset]
        prompt = self.model.build_prompt(prompt=msg_prompt, context=sys_instruction)
        logger.write("Prompt to update candidate based on edits:\n")
        logger.log_prompt(prompt)
        completion = await self.model.generate(prompt)

        return self.parse_completion(completion)

    def any_edits_suggested(self, completion):
        resp_lines = completion.split("\n")
        return any(line.startswith("- ") for line in resp_lines)

    async def check_with_validator_and_modify(
        self,
        logger: log.InstanceLogger,
        candidate: LinearProcedure,
        knowledge_str: str,
        max_updates: int = 3,
    ) -> list[str]:
        break_phrase = "NO UPDATE REQUIRED"
        updates_done = 0

        while updates_done < max_updates:
            validator_edits = await self.validate_update(logger, candidate)
            if break_phrase in validator_edits and not self.any_edits_suggested(validator_edits):
                logger.write(f"EXITING EDIT LOOP EARLY AFTER {updates_done} updates")
                break
            candidate.steps = await self.perform_validator_edits(
                logger, candidate, knowledge_str, validator_edits
            )
            logger.write("BEGIN EDITED STEPS\n")
            logger.write(textwrap.indent(candidate.format_steps(), "  ") + "\n")
            logger.write("END EDITED STEPS\n")
            updates_done += 1

        return candidate.steps
