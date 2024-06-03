from torchmetrics.text.rouge import ROUGEScore

from dataset import Procedure, format_steps
from evaluation.heuristic import Heuristic

class ROUGE_Score(Heuristic):
    def __init__(self):
        self.rouge = ROUGEScore(use_stemmer=True)
        
    def evaluate(self, gold: Procedure, generated: list[str]):
        # Input: Gold and Generated procedure steps
        print("Calculating rouge score")
        gold_steps = ""
        for i, proc_step in enumerate(gold.format_steps()):
            gold_steps += str(i) + ". "
            gold_steps += proc_step
            gold_steps += "\n"

        rouge_stats = self.rouge(format_steps(generated), gold_steps)
        for key in rouge_stats.keys():
            rouge_stats[key] = rouge_stats[key].item()
        print("Returning rouge score now")
        return rouge_stats