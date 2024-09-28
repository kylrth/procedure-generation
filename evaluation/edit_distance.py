import logging
import re

import editdistance as edt
import spacy
from langchain_core.messages import HumanMessage, SystemMessage

from dataset import LinearProcedure, format_steps
from systems import Model

from .heuristic import Heuristic


class EditDistance(Heuristic):
    model: Model

    def __init__(self, model: Model):
        self.model = model
        self.nlp = spacy.load("en_core_web_sm")

    async def get_one_word_list(self, proc_steps: str):
        prompt = [
            SystemMessage(
                content=(
                    "Please replace each step in the given procedure with a one-word description "
                    "of the action performed in that step. For example, 'mix flour and eggs in a "
                    "bowl' will be replaced with 'mixing'."
                )
            ),
            HumanMessage(
                content=proc_steps
                + "\nYour response should be a python list starting with [ and end with ]."
            ),
        ]

        one_word_list = await self.model.generate(prompt)
        one_word_list = one_word_list.replace("'", "")
        list_items = re.sub(r"[\s]", "", one_word_list)[1:-1].split(",")
        sentence = self.nlp(" ".join(list_items))
        # lemmatizing
        sentence = [
            word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_
            for word in sentence
        ]
        return sentence

    def evaluate(self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]):
        raise NotImplementedError

    async def aevaluate(
        self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]
    ) -> int:
        gold_proc_steps = gold.format_steps()  # Removing the brackets
        gold_one_word_list = await self.get_one_word_list(gold_proc_steps)
        logger.debug("Got one word list for gold")
        generated_one_word_list = await self.get_one_word_list(format_steps(generated))
        logger.debug("Got one word list for generated")
        edit_distance = edt.eval(generated_one_word_list, gold_one_word_list)
        logger.debug("Returning edit distance")
        return edit_distance
