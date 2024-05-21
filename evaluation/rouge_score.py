from torchmetrics.text.rouge import ROUGEScore

from dataset import Procedure


def compute_rouge_score(gold: Procedure, generated: str):
    # Input: Gold and Generated procedure steps
    rouge = ROUGEScore(use_stemmer=True)
    gold_steps = ""
    for i, proc_step in enumerate(gold.steps):
        gold_steps += str(i) + ". "
        gold_steps += proc_step
        gold_steps += "\n"

    rouge_stats = rouge(generated, gold_steps)
    return rouge_stats
