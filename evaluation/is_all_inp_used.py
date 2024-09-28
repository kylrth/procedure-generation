import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from dataset import LinearProcedure, format_steps
from systems import Model

from .heuristic import Heuristic


class AllInpUsed(Heuristic):
    model: Model
    dataset_name: str

    def __init__(self, model: Model, dataset_name: str):
        self.model = model
        self.dataset_name = dataset_name

    async def get_clean_input_list(self, inputs: str):
        filtered_sentence = []
        if self.dataset_name == "lcstep":
            resource_str_list = inputs.split(",")

            for w in resource_str_list:
                w = w.lower()
                if w.startswith("an"):
                    w = w[3:]
                elif w.startswith("a"):
                    w = w[2:]
                elif w.startswith("the"):
                    w = w[4:]
                filtered_sentence.append(w)

        elif self.dataset_name == "recipenlg":
            prompt = [
                SystemMessage(
                    content=(
                        "Please rephrase the ingredient list below keeping only the name of the "
                        "ingredient as it would appear in a recipe. For example, 8 oz. penne or "
                        "other tubular pasta becomes 'pasta', 1 1/2 c. frozen green beans becomes "
                        "'green beans', and 2 Tbsp. chopped flat leaf parsley becomes 'parsley'."
                    )
                ),
                HumanMessage(
                    content="Referring to above examples, modify the list below:\n"
                    + str(inputs)
                    + "\nYour response should be a python list starting with [ and end with ]. "
                    "Remove the text enclosed in ()."
                ),
            ]
            trim_ing_list = await self.model.generate(prompt)
            trim_ing_list = trim_ing_list.replace("'", "")
            filtered_sentence = re.sub(r"\[]", "", trim_ing_list)[1:-1].split(",")
        return filtered_sentence

    def evaluate(self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]):
        raise NotImplementedError

    async def aevaluate(
        self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]
    ) -> float:
        logger.debug("Getting cleaned input now")
        logger.debug(f"Input to procedure is: {gold.input_}")
        if gold.input_ is None or gold.input_ == "" or not isinstance(gold.input_, str):
            return -1
        filtered_inputs = await self.get_clean_input_list(gold.input_)
        logger.debug("Got cleaned inputs. Now checking!")
        if len(filtered_inputs) == 0:
            return 1
        formatted = format_steps(generated).lower()
        inp_found = 0
        for inp in filtered_inputs:
            par_idx = inp.find("(")
            if par_idx != -1:
                inp = inp[: par_idx - 1]
            inp = inp.strip().lower()
            if inp in formatted:
                inp_found += 1
        logger.debug("Returning all inputs used score")
        return inp_found / len(filtered_inputs)
