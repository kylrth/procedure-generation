import asyncio
from collections import defaultdict

from datasets import Dataset
import numpy as np

from evaluation.eval import evaluation
import recipenlg
from systems import Model, SystemInterface, ZeroShot


async def _evaluate(model, recipe):
    title = recipe["title"][0]
    ingredients = recipe["ingredients"][0]
    directions = recipe["directions"][0]
    res = await model.generate(title)

    scores = defaultdict(list)
    for completion in res:
        # if recipe was incorrectly generated
        if "Instructions" not in completion or "Ingredients" not in completion:
            continue

        evals = await evaluation(completion, recipenlg.format_recipe(ingredients, directions))
        for metric in evals:
            scores[metric].append(evals[metric])
    return scores, recipe["id"][0]


async def evaluate(model: SystemInterface, data: Dataset):
    tasks = [_evaluate(model, recipe) for recipe in data.iter(1)]
    results = await asyncio.gather(*tasks)

    # collect results
    scores = defaultdict(list)
    broken = []
    for evals, rid in results:
        # if recipe had no correct completions
        if len(evals) == 0:
            broken.append(rid)
            continue
        # collect all scores of a metric
        for metric, values in evals.items():
            for v in values:
                scores[metric].append(v)

    # average the results
    scores_avg = defaultdict()
    for metric, score in scores.items():
        scores_avg[metric] = np.mean(score)

    print(scores_avg)
    print("The model returned bad responses for these titles:")
    print("\n".join(str(i) for i in broken))


def main(model: str, data_dir: str = "./data"):
    model = Model.from_full_name(model)
    system = ZeroShot(model, "Please generate a recipe")

    data = recipenlg.load("val", data_dir).select(np.arange(0, 3))

    asyncio.run(evaluate(system, data))


# ignore (the dataset directory on my computer) C:\\Users\\mk_ya\\Desktop\\dataset\\dataset
if __name__ == "__main__":
    main("openai-gpt-3.5-turbo", "/mnt/leftie/data/recipenlg")
