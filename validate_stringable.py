"""Make sure all pickled graphs are loadable and stringable."""

import dataset
from graph import Graph


split_all = dataset.Split.TRAIN | dataset.Split.VAL | dataset.Split.TEST


if __name__ == "__main__":
    for const in [dataset.LCStep, dataset.RecipeNLG]:
        bad = set()
        ds = const("./dataset")
        graphs = ds.graphs(split_all)
        for i, g in enumerate(graphs):
            try:
                str(g)
            except (TypeError, Graph.DAGError):
                bad.add(i)
        if bad:
            bad_str = ", ".join(str(i) for i in bad)
            print(f"dataset had {len(bad)} bad graphs: {bad_str}")  # noqa: T201
