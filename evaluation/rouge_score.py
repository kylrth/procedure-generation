import logging

from torchmetrics.text.rouge import ROUGEScore

from dataset import LinearProcedure, format_steps

from .heuristic import Heuristic


class ScoreROUGE(Heuristic):
    def __init__(self):
        self.rouge = ROUGEScore(use_stemmer=True)

    def evaluate(
        self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]
    ) -> dict[str, float]:
        # Input: Gold and Generated procedure steps
        logger.debug("Calculating rouge score")
        gold_steps = ""
        for i, proc_step in enumerate(gold.format_steps()):
            gold_steps += str(i) + ". "
            gold_steps += proc_step
            gold_steps += "\n"

        rouge_stats = self.rouge(format_steps(generated), gold_steps)
        for key in rouge_stats:
            rouge_stats[key] = rouge_stats[key].item()
        logger.debug("Returning rouge score now")
        return rouge_stats

    async def aevaluate(self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]):
        raise NotImplementedError
