import logging

from dataset import LinearProcedure, format_steps

from .heuristic import Heuristic


class ApiOverlap(Heuristic):

    def get_apis(self, response: str):
        token_list = response.split("`")
        start_idx = 1
        api_list = []
        while start_idx < len(token_list):
            if "." in token_list[start_idx]:
                api_list.append(token_list[start_idx])
            start_idx += 2
        return api_list

    def evaluate(
        self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]
    ) -> float:
        gold_apis = self.get_apis(gold.format_steps())
        if not gold_apis:
            return 1  # nothing to check for
        logger.debug("Got gold apis")
        generated_apis = self.get_apis(format_steps(generated))
        logger.debug("Got generated apis")

        num_present = 0
        for api in gold_apis:
            if api in generated_apis:
                num_present += 1
        logger.debug("Returning API overlap score")
        return num_present / len(gold_apis)

    async def aevaluate(self, logger: logging.Logger, gold: LinearProcedure, generated: list[str]):
        raise NotImplementedError
