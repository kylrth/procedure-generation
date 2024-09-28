import random
from os import PathLike

import champ_dataset
import pickle
from .base import Dataset, Doc, LinearProcedure, GraphProcedure


def load_dataset():
    dataset = champ_dataset.load("v0")
    return dataset.problems, dataset.hints, dataset.concepts


def get_input(content, hints):
    input_str = ""

    # Adding category now
    input_str += f"Category: {content.category}\n"

    hints_list = []

    for elem in content.ch_list:
        if elem[0] == "H":
            hints_list.append(hints[elem]._text)

    # Adding hints now
    input_str += f"Hints: {hints_list}"

    return input_str


def get_solution_steps(content):
    steps_list = []
    for step in content.solution.steps:
        steps_list.append(step.text)

    steps_list.append(f"The answer is {content.answer}")
    return steps_list


def make_procedure_object(content, hints):
    input_str = get_input(content, hints)
    step_list = get_solution_steps(content)

    return LinearProcedure(input_str, content.text, step_list)


# Pass data dir as None for CHAMP dataset
class CHAMP(Dataset):
    def __init__(self, data_dir: str | PathLike, n: int | None = None):
        super().__init__(data_dir)
        self.n = n
        self.rng = random.Random(42)

    def _init_procedures(self) -> list[LinearProcedure]:
        probs, hints, _ = load_dataset()  # don't use concepts

        out = []
        for _, content in probs.items():
            proc_obj = make_procedure_object(content, hints)
            out.append(proc_obj)
        self.rng.shuffle(out)

        return out

    def _init_graphs(self) -> list[GraphProcedure]:
        dir = self.dir / "graphs" / "champ"
        file_list = dir.glob("*.pkl")
        graph_list = []
        for file in file_list:
            with file.open("rb") as f:
                graph = pickle.load(f)
            graph_list.append(graph)

        return graph_list

    def _get_docs(self) -> list[Doc]:
        doc_list = []
        _, _, concepts = load_dataset()
        for key, val in concepts.items():
            doc_list.append(Doc(title=key, contents=val._text))

        return doc_list
