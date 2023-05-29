from datasets import Dataset

import recipenlg
from systems import Model, SystemInterface, ZeroShot


def evaluate(model: SystemInterface, data: Dataset):
    for recipe in data.iter(1):
        title = recipe["title"][0]
        ingredients = recipe["ingredients"][0]
        directions = recipe["directions"][0]

        res = model.generate(title)

        # TODO evaluate


def main(model: str, data_dir: str = "./data"):
    model = Model.from_full_name(model)
    system = ZeroShot(model, "Please generate a recipe")

    data = recipenlg.load("val", data_dir)

    evaluate(system, data)
