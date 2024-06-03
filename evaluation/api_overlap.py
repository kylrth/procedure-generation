from evaluation.heuristic import Heuristic
from dataset import Procedure, format_steps

class Api_Overlap(Heuristic):
    
    def get_apis(self, response: str):
        token_list = response.split("`")
        start_idx = 1
        api_list = []
        while start_idx < len(token_list):
            if '.' in token_list[start_idx]:
                api_list.append(token_list[start_idx])
            start_idx += 2
        # import pdb; pdb.set_trace()
        return api_list


    def evaluate(self, gold: Procedure, generated: list[str]):
        gold_apis = self.get_apis(gold.format_steps())
        print("Got gold apis")
        generated_apis = self.get_apis(format_steps(generated))
        print("Got generated apis")

        num_present = 0
        for api in gold_apis:
            if api in generated_apis:
                num_present += 1
        print("Returning API overlap score")
        return num_present / len(gold_apis)
