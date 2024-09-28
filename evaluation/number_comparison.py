import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from dataset import LinearProcedure, format_steps
from systems import Model

from .heuristic import Heuristic


class NumberComparison(Heuristic):
    model: Model

    def __init__(self, model: Model):
        self.model = model

    async def get_numbers_list(self, proc_steps: str):

        prompt = [
            SystemMessage(
                content=(
                    "Please extract the numbers along with their units only from the procedure "
                    "given below. For example, 'preheat the oven to 350°' gives '350° C'. However, "
                    "do not extract numbers which are description of the crockery and also avoid "
                    "repetition of numbers in your response.\n"
                )
            ),
            HumanMessage(
                content=proc_steps
                + "\nYour response should be a python list starting with [ and end with ]."
            ),
        ]

        number_list = await self.model.generate(prompt)
        number_list = number_list.replace("'", "")
        list_items = re.sub(r"\[]", "", number_list)[1:-1].split(",")
        return list_items

    def evaluate(self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]):
        raise NotImplementedError

    async def aevaluate(self, _, gold: LinearProcedure, generated: list[str]) -> float:
        gold_proc_steps = gold.format_steps()  # Removing the brackets
        joined = format_steps(generated)
        gold_numbers_list = await self.get_numbers_list(gold_proc_steps)
        generated_numbers_list = await self.get_numbers_list(joined)

        gold_numbers_list = [x.strip().lower() for x in gold_numbers_list]
        generated_numbers_list = [x.strip().lower() for x in generated_numbers_list]

        num_present = 0
        for number in gold_numbers_list:
            if number in generated_numbers_list:
                num_present += 1

        return num_present / len(gold_numbers_list)
