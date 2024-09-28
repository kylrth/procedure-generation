import textwrap
from typing import ClassVar

import retrieval
from dataset import LinearProcedure
from utils import log

from .few_shot import FewShot
from .interface import Response
from .model import Model


class RAG(FewShot):
    """This model prompts an LM to generate the response few-shot, but with the examples provided by
    searching a vector store for texts with similar embeddings to the query."""

    model: Model
    docs: retrieval.DocStore
    k: int
    critic: bool

    def __init__(self, model: Model, docs: retrieval.DocStore, k: int, dataset: str, critic: bool):
        super().__init__(model, dataset)
        self.docs = docs
        self.k = k
        self.critic = critic

    async def generate(self, logger: log.InstanceLogger, query: str, input_: str) -> Response:
        # retrieve docs
        docs = await self.docs.search(f"{query} using {input_}", self.k)
        logger.write(f"retrieved {len(docs)} docs\n")

        # build prompt
        context = self.build_context(docs)
        msg_prompt = (
            context + "\n\n" + self._prompt_inst[self.dataset].format(query=query, input_=input_)
        )
        prompt = self.model.build_prompt(msg_prompt, self._instructions[self.dataset])
        logger.write(f"prompt to model {self.model.name}:\n")
        logger.log_prompt(prompt)

        # call model
        completion = await self.model.generate(prompt)
        res = self._make_result(prompt, completion)

        if not self.critic:
            return res

        # call critic
        res.answer = await self.check_with_validator_and_modify(
            logger, LinearProcedure(input_, query, res.answer), context, max_updates=3
        )

        res.input_tokens = -1
        res.output_tokens = -1

        return res

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
            if break_phrase in validator_edits:
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
            "You should always suggest only your edits in a bulleted list. If there "
            "are no edits to be made, please only respond 'NO UPDATE REQUIRED'. You are "
            "required to strictly follow the mentioned output format."
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

    _perform_edits_instructions: ClassVar[dict[str, str]] = {
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

        sys_instruction = self._perform_edits_instructions[self.dataset]
        prompt = self.model.build_prompt(prompt=msg_prompt, context=sys_instruction)
        logger.write("Prompt to update candidate based on edits:\n")
        logger.log_prompt(prompt)
        completion = await self.model.generate(prompt)

        return self.parse_completion(completion)
