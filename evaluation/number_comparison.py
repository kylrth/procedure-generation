import re

from langchain.schema import HumanMessage, SystemMessage
import asyncio
from dataset import Procedure, format_steps
from systems import Model
from evaluation.heuristic import Heuristic

class Number_Comparison(Heuristic):
    model: Model
    
    def __init__(self, model: Model):
        self.model = model
    
    async def get_numbers_list(self, proc_steps: str):
        
        # prompt = [
        #     SystemMessage(
        #         content=("[INSTRUCTION]\
        #             You are a chef whose role is to validate the temperatures, cooking times and the proportions of ingredients mentioned in the given recipe. To do so, please extract the numbers along with their units only from the procedure given below. For example, 'preheat the oven to 350°' gives '350° C' and '2 thinly sliced red onions' gives '2 onions'. However, do not extract numbers which are description of the crockery and avoid repetition of numbers in your response."
        #         )
        #     ),
        #     HumanMessage(
        #         content=f"\n\n[BEGIN PROCEDURE]\n{proc_steps}\n[END PROCEDURE]\n\nYour response should be a python list starting with [ and end with ]."
        #     ),
        # ]
        
        prompt = [
            SystemMessage(
                content=(
                    "Please extract the numbers along with their units only from the procedure given "
                    "below. For example, 'preheat the oven to 350°' gives '350° C'. However, do not "
                    "extract numbers which are description of the crockery and also avoid repetition "
                    "of numbers in your response.\n"
                )
            ),
            HumanMessage(
                content=proc_steps
                + "\nYour response should be a python list starting with [ and end with ]."
            ),
        ]

        number_list = (await self.model.agenerate(prompt))[0]
        number_list = number_list.replace("'", "")
        list_items = re.sub(r"\[]", "", number_list)[1:-1].split(",")
        # import pdb; pdb.set_trace()
        return list_items

    async def aevaluate(self, gold: Procedure, generated: list[str]):
        gold_proc_steps = gold.format_steps()  # Removing the brackets
        generated = format_steps(generated)
        gold_numbers_list = await self.get_numbers_list(gold_proc_steps)
        generated_numbers_list = await self.get_numbers_list(generated)
        
        gold_numbers_list = [x.strip().lower() for x in gold_numbers_list]
        generated_numbers_list = [x.strip().lower() for x in generated_numbers_list]
        
        num_present = 0
        for number in gold_numbers_list:
            if number in generated_numbers_list:
                num_present += 1
        
        return num_present / len(gold_numbers_list)