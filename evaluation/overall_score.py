import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from dataset import LinearProcedure, format_steps
from model import Model

from .heuristic import Heuristic


class OverallScore(Heuristic):
    model: Model

    def __init__(self, model: Model):
        self.model = model

    def prepare_prompt(self, gold_steps: str, proc_steps: str, output: str):
        prompt = [
            SystemMessage(
                content=(
                    "[Instruction] Please act as an impartial judge and evaluate the quality of "
                    "the procedure provided by an AI assistant to achieve the user goal displayed "
                    "below. For this evaluation, you should primarily evaluate if the assistant "
                    "answer provides the same answer and information as the following ground "
                    "truth. If the ground truth provides more information than the assistant "
                    "answer, it should be penalized; if the ground truth provides information "
                    "which is not relevant to the user goal than the assistant answer, it should "
                    "not be penalized; if the assistant answer provides more information, you "
                    "should evaluate if this information is relevant and correct. Do not penalize "
                    "differences in the sentences' structure, grammar, wording, but focus on the "
                    "facts and suggestions being made by the assistant; if the assistant answer "
                    "leads to the same user interpretation and action than the ground truth, then "
                    "the score should be high. However, if key information (like a number "
                    "referring to an identifier, amount, names, etc.) that is found in the ground "
                    "truth differs in the assistant answer, it should be penalized, mostly if it "
                    "changes how the answer is interpreted, wrt the question. Compare the "
                    "assistant answer and the Ground truth, then provide a short explanation on "
                    "your analysis of the assistant answer quality based on the comparison. Be as "
                    "objective as possible. Based on your explanation, you must rate the response "
                    'on a scale of 0 to 10 by strictly following this format: "[[rating]]", for '
                    'example: "Rating: [[5]]". Print this rating at the END only. As a guideline '
                    "for scoring:\n\n"
                    "[[0]]: the answer is completely contrary to the ground truth and is "
                    "incorrect.\n"
                    "[[2]]: the answer largely differs from the ground truth, the important "
                    "elements found in the ground truth are not present in the assistant answer.\n"
                    "[[5]]: the answer does not provide all the elements found in the ground truth "
                    "and some of the elements added by the assistant answer may be incorrect.\n"
                    "[[8]]: the answer provides most of the elements found in the ground truth, "
                    "the differences are mainly wording but represent the same facts and "
                    "suggestions. The differences are relevant and correct.\n"
                    "[[10]]: the answer is the same as the ground truth. There may be very small "
                    "wording differences but they mean exactly the same and use the exact same "
                    "facts and references."
                )
            ),
            HumanMessage(
                content=(
                    f"[User Goal]\nProvide a procedure to accomplish: {output}\n\n"
                    "[The Start of the Ground truth]\n"
                    f"{gold_steps}\n"
                    f"[The End of the Ground truth]\n\n"
                    "[Assistant Answer]\n"
                    f"{proc_steps}\n\n"
                    'Your answer should begin with "Here is my analysis of the comparison between '
                    "ground truth and the assistant's answer:\n"
                    '1."'
                )
            ),
        ]

        return prompt

    def result_parser(self, sentence):
        if not sentence:
            return -2
        match = re.findall(r"\s*\[\[(\d+)\]\]\s*$", sentence)
        if not match:
            match = re.findall(r"\s*\[\[(\d+)\]\]\s*", sentence)
            if not match:
                match = re.search(r"\s*(N\/A)\s*$", sentence)
                if not match:
                    return -3
                return 1

        return int(match[-1])

    def evaluate(self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]):
        raise NotImplementedError

    async def aevaluate(
        self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]
    ) -> int:
        prompt = self.prepare_prompt(gold.format_steps(), format_steps(generated), gold.output)
        logger.debug("Prompt prepared")
        answer = await self.model.generate(prompt)
        if answer.find("[") == -1:
            logger.debug("Generating the response again")
            answer = await self.model.generate(prompt)
        logger.debug("Got model response for overall score")
        score = self.result_parser(answer)
        logger.debug("Returning the achieved score!")
        return score
