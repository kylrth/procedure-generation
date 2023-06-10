import logging
from os import PathLike
from pathlib import Path

import numpy as np
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS

import recipenlg

from .interface import System
from .model import Model, log
from .zero_shot import ZeroShot


class FewShot(System):
    """This model prompts an LM to generate text few-shot, with the examples provided by searching a
    vector store for texts with similar embeddings to the title."""

    instructions: str = ZeroShot.instructions

    def __init__(
        self,
        model: Model,
        k: int,
        embedding_model: str,
        vs_path: str | PathLike = "vectors",
        emb_path: str | PathLike = "embeddings",
        data_dir: str | PathLike = "data",
    ):
        self.model = model

        vs_path = Path(vs_path)
        if vs_path.exists():
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model, encode_kwargs={"device": "cpu"}
            )
            store = FAISS.load_local(vs_path, embeddings)
        else:
            ds = recipenlg.load("train", data_dir)
            emb_path = Path(emb_path) / embedding_model
            embeds = np.concatenate(np.loadtxt(file) for file in emb_path.glob("*.txt.gz"))

            # TODO remove this
            keep = 100000
            ds = ds.select(np.arange(0, keep))
            embeds = embeds[:keep]

            store = FAISS.from_embeddings(zip(ds["formatted"], embeds, strict=True))
            store.save_local(vs_path)

        self.selector = SemanticSimilarityExampleSelector(vectorstore=store, k=k)

    def generate(self, title: str, logger: logging.Logger | None = None) -> list[str]:
        examples = self.get_examples(title)
        prompt = self.model.build_prompt(title, self.instructions, examples)
        completion = self.model.generate(prompt)

        log(logger, "FewShot", prompt, completion)

        return completion

    async def agenerate(self, title: str, logger: logging.Logger | None = None) -> list[str]:
        examples = self.get_examples(title)
        prompt = self.model.build_prompt(title, self.instructions, examples)
        completion = await self.model.agenerate(prompt)

        log(logger, "FewShot", prompt, completion)

        return completion

    def get_examples(
        self, title: str, logger: logging.Logger | None = None
    ) -> list[tuple[str, str]]:
        """Returns the examples that will be inserted into the prompt."""
        examples = self.selector.select_examples({"title": title})
        examples = [(example["title"], example["recipe"]) for example in examples]

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
                    logger.warning(
                        f"keeping only {i}/{len(examples)} examples for few-shot prompt due to "
                        "token length"
                    )
                    examples = examples[:i]
                    break

        return examples
