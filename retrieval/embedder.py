import pickle
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Type

import mmh3
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer


class Embedder(ABC):
    """Something that produces embeddings."""

    @abstractmethod
    async def embed(self, text: list[str]) -> list[np.ndarray]:
        pass


class NamedEmbedder(Embedder):
    """An embedder that can be constructed by name."""

    @abstractmethod
    def __init__(self, model: str):
        pass


class OpenAIEmbedder(NamedEmbedder):
    model: str
    client: AsyncOpenAI

    def __init__(self, model: str):
        self.model = model
        self.client = AsyncOpenAI()

    async def embed(self, text: list[str]) -> list[np.ndarray]:
        response = await self.client.embeddings.create(input=text, model=self.model)

        return [np.array(embed.embedding) for embed in response.data]


class HFEmbedder(NamedEmbedder):
    model: SentenceTransformer

    def __init__(self, model: str):
        if model == "all-mpnet-base-v2":
            self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        else:
            raise NotImplementedError(model)

    async def embed(self, text: list[str]) -> list[np.ndarray]:
        return self.model.encode(text, show_progress_bar=len(text) > 1, convert_to_numpy=True)


embedder_dict: dict[str, Type[NamedEmbedder]] = {
    "hf": HFEmbedder,
    "openai": OpenAIEmbedder,
}


def embedder_from_name(name: str) -> NamedEmbedder:
    service, model = name.split("-", maxsplit=1)

    try:
        return embedder_dict[service](model)
    except KeyError:
        raise NotImplementedError(name) from None


class CachingEmbedder(Embedder):
    """Caches all embeddings in a disk-backed cache."""

    e: Embedder
    path: Path
    cache: dict[str, np.ndarray]

    def __init__(self, e: Embedder, path: str | PathLike):
        self.e = e
        self.path = Path(path)
        self.cache = {}

    async def embed(self, text: list[str]) -> list[np.ndarray]:
        """First searches the in-memory cache, then the disk cache, and otherwise calls e.embed."""
        out = []
        to_embed: list[str] = []
        for s in text:
            # in-memory cache
            if s in self.cache:
                out.append(self.cache[s])
                continue

            # disk cache
            try:
                embed = self._read_embed(s)[s]
                out.append(embed)
                continue
            except FileNotFoundError:
                pass

            out.append(len(to_embed))  # track the index of the embedding for when we generate
            to_embed.append(s)

        # generate any embeddings not cached
        if len(to_embed) > 0:
            embeds = await self.e.embed(to_embed)
            for i in range(len(out)):
                if isinstance(out[i], int):
                    embed = embeds[out[i]]
                    out[i] = embed
                    self.cache[text[i]] = embed

        return out

    def _filepath(self, text: str) -> Path:
        h = hex(mmh3.hash128(text, signed=False))

        return self.path / (h + ".pkl")

    def _read_embed(self, text: str) -> dict[str, np.ndarray]:
        with self._filepath(text).open("rb") as f:
            return pickle.load(f)

    def _write_embed(self, text: str, embed: np.ndarray):
        try:
            d = self._read_embed(text)
        except FileNotFoundError:
            d = {}

        d[text] = embed

        with self._filepath(text).open("wb") as f:
            pickle.dump(d, f)

    def flush(self):
        """Write all cached embeddings to disk and empty the in-memory cache."""
        self.path.mkdir(parents=True, exist_ok=True)

        for k, v in self.cache.items():
            self._write_embed(k, v)

        self.cache.clear()

    def load_all(self):
        """Fill the in-memory cache with every entry on disk.

        This will overwrite any embeddings already in the in-memory cache if the text was the same.
        """
        for filepath in self.path.glob("*.pkl"):
            with filepath.open("rb") as f:
                for k, v in pickle.load(f).items():
                    self.cache[k] = v
