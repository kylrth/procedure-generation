import logging
from os import PathLike
from pathlib import Path

import numpy as np
from datasets import Dataset
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS

from .few_shot import FewShot
from .interface import System
from .model import Model, log


def load_embeddings(path: str | PathLike, max_rows: int | None = None) -> np.ndarray:
    """Use np.loadtxt to read each embedding file in path until the concatenated result has max_rows
    rows."""
    tot = 0
    arrays = []

    i = 1
    while True:
        try:
            array = np.loadtxt(
                Path(path) / f"{i}.txt.gz",
                max_rows=max_rows - tot if max_rows is not None else None,
            )
        except FileNotFoundError:
            if len(arrays) == 0:
                # the embeddings are missing
                raise
            break

        arrays.append(array)
        tot += len(array)

        if max_rows and tot >= max_rows:
            break

        i += 1

    return np.concatenate(arrays)


class RAG(System):
    """This model prompts an LM to generate text few-shot, with the examples provided by searching a
    vector store for texts with similar embeddings to the title."""

    instructions: str = FewShot.instructions

    def __init__(
        self,
        model: Model,
        k: int,
        ds: Dataset,
        embedder: HuggingFaceEmbeddings,
        embedding_n: int | None = None,
        emb_path: str | PathLike = "cache/embeddings",
        vs_path: str | PathLike = "cache/vectors",
    ):
        self.model = model

        vs_path = Path(vs_path)
        if vs_path.exists():
            store = FAISS.load_local(str(vs_path), embedder)
        else:
            embeds = load_embeddings(emb_path, max_rows=embedding_n)

            ds = ds.select(np.arange(0, len(embeds)))

            # We would use FAISS.from_embeddings, but it has a stupid list[tuple[str, list[float]]]
            # argument.
            store = FAISS._FAISS__from(
                texts=ds["formatted"],
                embeddings=embeds,
                embedding=embedder,
                metadatas=ds,
                ids=list(map(str, ds["id"])),
            )
            store.save_local(vs_path)

        self.selector = SemanticSimilarityExampleSelector(vectorstore=store, k=k)

    def generate(self, title: str, logger: logging.Logger | None = None) -> list[str]:
        examples = self.get_examples(title, logger)
        prompt = self.model.build_prompt(title, self.instructions, examples)
        completion = self.model.generate(prompt)

        log(logger, "FewShot", prompt, completion)

        return completion

    async def agenerate(self, title: str, logger: logging.Logger | None = None) -> list[str]:
        examples = self.get_examples(title, logger)
        prompt = self.model.build_prompt(title, self.instructions, examples)
        completion = await self.model.agenerate(prompt)

        log(logger, "FewShot", prompt, completion)

        return completion

    def get_examples(
        self, title: str, logger: logging.Logger | None = None
    ) -> list[tuple[str, str]]:
        """Returns the examples that will be inserted into the prompt."""
        examples = self.selector.select_examples({"title": title})
        for i in range(len(examples)):
            # split the title from the recipe
            recipe = examples[i]["formatted"].split("\n\n", 1)[1]
            examples[i] = (examples[i]["title"], recipe)

        if logger:
            logger.debug(f"got {len(examples)} examples: {', '.join(ex[0] for ex in examples)}")

        token_budget = self.model.max_tokens
        if token_budget is not None:
            token_counts = [self.model.get_num_tokens(example) for example in examples]

            # As a heuristic, we'll assume the remaining token window for inference only needs to be
            # about as large as the largest example we provide. I'd imagine that similar recipes
            # require similar lengths.
            token_budget -= max(token_counts) + 20

            for i, token_count in enumerate(token_counts):
                token_budget -= token_count
                if token_budget < 0:
                    if logger:
                        logger.warning(
                            f"keeping only {i}/{len(examples)} examples for few-shot prompt due to "
                            "token length"
                        )
                    examples = examples[:i]
                    break

        return examples
