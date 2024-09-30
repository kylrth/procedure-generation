import random
import re
import sys
import textwrap
from typing import ClassVar

from dataset import GraphProcedure, LinearProcedure, create_graphs_for_graph_store
from model import Model
from retrieval import GraphProcedureStore
from utils import log
from utils.workers import spread_gather

from .interface import Response, System


class AAG(System):
    model: Model
    skills: GraphProcedureStore
    k: int
    dataset: str
    summarize: bool
    use_critic: bool
    hs: bool

    def __init__(
        self,
        model: Model,
        skills: GraphProcedureStore,
        k: int,
        dataset: str,
        critic: bool,
        hs: bool,
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
        self.use_critic = critic
        self.n_queries = n_queries
        self.hs = hs

    async def generate(self, logger: log.InstanceLogger, query: str, input_: str) -> Response:
        queries = await self.queries_relevant_to(logger, query, input_)
        answers = await self.get_answers_to_queries(logger, queries)
        knowledge_str = self.format_knowledge(queries, answers)

        candidate_steps = await self.update_steps(logger, None, input_, knowledge_str)
        candidate_linear = LinearProcedure(input_, query, candidate_steps)
        candidate = await create_graphs_for_graph_store(
            logger, -1, candidate_linear, self.model, self.dataset, save_pkl=False
        )

        if self.use_critic:
            max_critic_cycles = 3
            valid = False
            while max_critic_cycles > 0 and not valid:
                questions = await self.critic(logger, query, input_, candidate)
                if len(questions) == 0:
                    valid = True
                    continue
                answers = await self.get_answers_to_queries(logger, questions)
                knowledge_str = self.format_knowledge(queries, answers)
                candidate_steps = await self.update_steps(logger, candidate, input_, knowledge_str)
                candidate_linear = LinearProcedure(input_, query, candidate_steps)
                candidate = await create_graphs_for_graph_store(
                    logger, -1, candidate_linear, self.model, self.dataset, save_pkl=False
                )
                max_critic_cycles -= 1
        return Response(
            answer=candidate,
            model=self.model.name,
            # TODO token counts
        )

    async def get_answers_to_queries(
        self, logger: log.InstanceLogger, queries: list[str]
    ) -> list[str]:
        procs: list[list[GraphProcedure]] = []
        answers: list[str] = []
        for q in queries:
            if self.hs:
                proc_list = await self.skills.hierarchical_retrieval(q, k=2 * self.k, k2=self.k)
            else:
                proc_list = await self.skills.search(q, k=self.k)
            procs.append(proc_list)
            answers.append(await self.answer_the_question(logger, q, proc_list))

        logger.write(f"answered {len(queries)} search queries from {self.model.name}:\n")
        for i, q in enumerate(queries):
            logger.write(f"- {q}\n")
            for p in procs[i]:
                logger.write(f"  - {p.get_title()}\n")

        return answers

    _instructions_with_cand: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please update the provided list of steps to accomplish the specified goal using "
            "the LangChain Python library. Don't include code, extraneous commentary, or examples, "
            "but do refer to the specific LangChain APIs (or other APIs) used in each step. "
            "Incorporate information from any of the provided reference question and answers to "
            "refine the steps to achieve the specified goal."
        ),
        "recipenlg": (
            "Please update the provided list of steps to create a recipe for the specified "
            "food. Don't include extraneous commentary, or examples, but do refer to the special "
            "characteristics and state of the ingredients used in each step. "
            "Incorporate information from any of the provided reference question and answers to "
            "refine the steps to achieve the specified goal."
        ),
        "champ": (
            "Please update the provided list of steps to solve the given math problem. "
            "Don't include code, extraneous commentary, or examples, but do refer to the concepts "
            "and hints used in each step. Incorporate information from any of the provided "
            "reference question and answers to refine the steps to achieve the specified goal."
        ),
    }

    _prompt_inst_with_cand: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Note that each step under 'Steps' is a key value pair where key is of the format:\n"
            "(<list of step inputs>) -> <step output> and value is the natural language description"
            " of the step.\n\n"
            "Please update the candidate steps to accomplish '{query}' using the knowledge "
            "above. Create and use these resources in your response: {input_}. "
            "Please output only the updated steps in natural language (like the values above). "
            "Your response should start with '1.'. "
            "The final response should not contain direct references to specific titles "
            "in the knowledge above."
        ),
        "recipenlg": (
            "Note that each step under 'Steps' is a key value pair where key is of the format:\n"
            "(<list of step inputs>) -> <step output> and value is the natural language description"
            " of the step.\n\n"
            "Please update the list of steps to accomplish '{query}' using the knowledge "
            "above. Use these ingredients in your response: {input_}. "
            "Please output only the updated steps in natural language (like the values above). "
            "Your response should start with '1.'. "
            "The final response should not contain direct references to specific titles "
            "in the knowledge above."
        ),
        "champ": (
            "Note that each step under 'Steps' is a key value pair where key is of the format:\n"
            "(<list of step inputs>) -> <step output> and value is the natural language description"
            " of the step.\n\n"
            "Please update the list of steps to solve '{query}' using the knowledge above. "
            "Use this additional information in preparing your response: {input_}. "
            "Please output only the updated steps in natural language (like the values above). "
            "Your response should start with '1.'. "
            "The final response should not contain direct references to specific titles "
            "in the knowledge above."
        ),
    }

    _instructions: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please generate a list of steps to accomplish the specified goal using the LangChain "
            "Python library. Don't include code, extraneous commentary, or examples, "
            "but do refer to the specific LangChain APIs (or other APIs) used in each step. "
            "Use information from any of the provided reference question and answers to formulate "
            "the steps to achieve the specified goal."
        ),
        "recipenlg": (
            "Please generate a list of steps to create a recipe for the specified food. "
            "Don't include extraneous commentary, or examples, but do refer to the special "
            "characteristics and state of the ingredients used in each step. Use information "
            "from any of the provided reference question and answers to formulate the steps to "
            "achieve the specified goal."
        ),
        "champ": (
            "Please generate a list of steps to solve the given math problem. Don't include code, "
            "extraneous commentary, or examples, but do refer to the concepts and hints used in "
            "each step. Use information from any of the provided reference question and answers to "
            "formulate the steps to achieve the specified goal."
        ),
    }

    _prompt_inst: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please provide the list of steps to accomplish '{query}' using the knowledge "
            "above. Create and use these resources in your response: {input_}. "
            "Don't produce any text other than the list of steps. Your response should start "
            "with '1.'. The final response should not contain direct references to specific titles "
            "in the knowledge above."
        ),
        "recipenlg": (
            "Please provide the list of steps to accomplish '{query}' using the knowledge "
            "above. Use these ingredients in your response: {input_}. "
            "Don't produce any text other than the list of steps. Your response should start "
            "with '1.'. The final response should not contain direct references to specific titles "
            "in the knowledge above."
        ),
        "champ": (
            "Please provide the list of steps to solve '{query}' using the knowledge above. "
            "Use this additional information in preparing your response: {input_}. "
            "Don't produce any text other than the list of steps. Your response should start "
            "with '1.'. The final response should not contain direct references to specific titles "
            "in the knowledge above."
        ),
    }

    def format_knowledge(self, queries: list[str], summaries: list[str]) -> str:
        sum_str = ""
        for q, summary in zip(queries, summaries):
            sum_str += f"Q: {q}\nA: {summary}\n\n"

        return sum_str.rstrip()

    async def update_steps(
        self,
        logger: log.InstanceLogger,
        candidate: GraphProcedure | None,
        input_: str,
        knowledge_str: str,
    ) -> list[str]:
        if candidate is not None:
            msg_prompt = (
                f"[BEGIN KNOWLEDGE]\n{knowledge_str}\n[END KNOWLEDGE]"
                "\n\n"
                f"[BEGIN CANDIDATE]\n{str(candidate)}\n[END CANDIDATE]"
                "\n\n"
                + +self._prompt_inst_with_cand[self.dataset].format(
                    query=candidate.get_title(), input_=input_
                )
            )
            sys_instruction = self._instructions_with_cand[self.dataset]
        else:
            msg_prompt = (
                f"[BEGIN KNOWLEDGE]\n{knowledge_str}\n[END KNOWLEDGE]"
                "\n\n"
                + self._prompt_inst[self.dataset].format(query=candidate.get_title(), input_=input_)
            )
            sys_instruction = self._instructions[self.dataset]

        prompt = self.model.build_prompt(prompt=msg_prompt, context=sys_instruction)
        logger.write("Prompt to update candidate steps:\n")
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

    def format_procedures(self, procs: list[GraphProcedure]) -> str:
        seen = set()
        out = ""
        for p in procs:
            if p in seen:
                continue
            out += f"\n\n{self._example_name[self.dataset]} {p!s}"
            seen.add(p)

        return out[2:]  # skip first "\n\n"

    _llm_roles: ClassVar[dict[str, str]] = {
        "recipenlg": "recipes",
        "lcstep": "programming with LangChain library",
        "champ": "solving maths problems",
    }

    async def answer_the_question(
        self, logger: log.InstanceLogger, query: str, procs: list[GraphProcedure]
    ) -> str:
        context = self.format_procedures(procs)
        sys_instruction = (
            "[Instruction]\n"
            "You are an expert at {role} whose job is to use the "
            "information provided below to answer the question. Please include the "
            "information only from the provided knowledge and make sure "
            "that the answer is complete, short and concise. Avoid introductory and "
            "closing lines at the start and end of your response. "
            "Don't directly refer to the titles in the provided knowledge when generating the "
            "answer."
        )
        msg_prompt = (
            f"[BEGIN QUESTION]\n{query}\n[END QUESTION]\n\n"
            f"[BEGIN INFORMATION]\n{context}\n[END INFORMATION]\n\n"
            "Note that each step under 'Steps' is a key value pair where key is of the format:\n"
            "(<list of step inputs>) -> <step output> and value is the natural language description"
            " of the step."
        )
        prompt = self.model.build_prompt(
            prompt=msg_prompt,  # Human Message
            context=sys_instruction.format(role=self._llm_roles[self.dataset]),  # System Message
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

    _critic_sys: ClassVar[str] = (
        "You are {role} tasked with critiquing the provided procedure. Please list up to 4 "
        "criticisms in a simple bulleted list. Don't criticize the goal or inputs, as these are "
        "requirements. Be conservative and only point out real problems. If there are no real "
        "issues, say NO CRITIQUES."
    )
    _critic_role: ClassVar[dict[str, str]] = {
        "lcstep": "an expert in developing LangChain applications in Python",
        "recipenlg": "an expert chef",
        "champ": "a skill competition math problem solver",
    }

    _critic_react: ClassVar[str] = (
        "You are preparing a {p_name} for '{query}' using the following {i_name_pl}:\n"
        "{inputs}\n"
        "Here's what you wrote:\n"
        "\n"
        "{candidate}\n"
        "\n"
        "A reviewer says the following:\n"
        "\n"
        "{criticism}\n"
        "\n"
        "Please state whether you agree with this criticism, and why. Then, if you disagree, write "
        "'Result: WON'T FIX'. If you agree, write instead a search question to help you discover "
        "the information you need to solve the problem, like 'Result: <question>'."
    )
    _p_name: ClassVar[dict[str, str]] = {
        "lcstep": "step-by-step procedure",
        "recipenlg": "recipe",
        "champ": "step-by-step solution",
    }
    _i_name_pl: ClassVar[dict[str, str]] = {
        "lcstep": "resources",
        "recipenlg": "ingredients/tools",
        "champ": "hints",
    }

    async def critic(
        self, logger: log.InstanceLogger, query: str, input_: str, candidate: GraphProcedure
    ) -> list[str]:
        """Critique the candidate procedure and produce a list of questions to get the necessary
        information to fix the issues."""
        # collect criticism
        sys = self._critic_sys.format(role=self._critic_role[self.dataset])
        hum = str(candidate)
        res = await self.model.generate(self.model.build_prompt(hum, context=sys))
        if "NO CRITIQUES" in res and "\n- " not in res:
            logger.write("NO CRITIQUES\n")
            return []
        criticisms = [crit[2:].strip() for crit in res.strip().split("\n") if crit.startswith("- ")]

        # convert criticisms to queries
        async def _task(criticism):
            prompt = self._critic_react.format(
                p_name=self._p_name[self.dataset],
                query=query,
                i_name_pl=self._i_name_pl[self.dataset],
                inputs=input_,
                candidate=str(candidate),
                criticism=criticism,
            )

            return criticism, await self.model.generate(self.model.build_prompt(prompt))

        results = await spread_gather(_task, criticisms, n=5)
        search_queries = []
        logger.write("BEGIN CRITICISMS\n")
        for criticism, completion in results:
            if "Result: WON'T FIX" in completion:
                logger.write(f"  - {criticism}: WON'T FIX\n")
                continue

            search_query = completion.rsplit("Result: ", 1)[1]
            logger.write(f"  - {criticism}: '{search_query}'\n")
            search_queries.append(search_query)
        logger.write("END CRITICISMS\n")

        return search_queries
