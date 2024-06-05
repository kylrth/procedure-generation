import os
import pickle
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Generator, Iterable, Type, TypeVar

import mmh3
import numpy as np
import tiktoken
import tqdm
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from utils import spread_gather


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


_openai_max_input: dict[str, int] = {
    # https://platform.openai.com/docs/guides/embeddings/embedding-models
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
}


T = TypeVar("T")


def batch(tokens: Iterable[list[T]], max_tok: int) -> Generator[list[list[T]], None, None]:
    """Yield batches of sublists which in total are not longer than max_tok."""
    current_batch = []
    current_length = 0

    for seq in tokens:
        if len(seq) > max_tok:
            raise ValueError(f"sequence of length {len(seq)} is too long (> {max_tok})")

        if current_length + len(seq) > max_tok:
            yield current_batch

            current_batch = [seq]
            current_length = len(seq)
        else:
            current_batch.append(seq)
            current_length += len(seq)

    if current_batch:
        # yield the remaining batch
        yield current_batch


class OpenAIEmbedder(NamedEmbedder):
    model: str
    client: AsyncOpenAI
    tok: tiktoken.Encoding
    _tok_threads: int
    max_input: int

    def __init__(self, model: str):
        self.model = model
        self.client = AsyncOpenAI()
        self.tok = tiktoken.encoding_for_model(model)
        self._tok_threads = min(32, os.cpu_count() or 1)
        self._max_input = _openai_max_input[model]

    def _tokenize(self, text: list[str]) -> Generator[list[int], None, None]:
        """Tokenize the strings.

        For short lists of texts, this generates token sequences on demand. For long lists of texts,
        tokenization happens all at once with a thread pool before the first item is yielded."""
        if len(text) < self._tok_threads * 4:
            # It's probably not worth using threads for so few texts
            for s in text:
                yield self.tok.encode(s)
        else:
            yield from self.tok.encode_batch(text, num_threads=self._tok_threads)

    async def embed(self, text: list[str]) -> list[np.ndarray]:
        tokens = self._tokenize(text)
        batches = batch(tokens, self._max_input)

        results = {}

        with tqdm.tqdm(total=len(text), disable=len(text) <= 1) as pbar:
            # the async task is to fetch the embeddings for a batch of token sequences, update the
            # status bar, and return embeddings as arrays
            async def task(args: tuple[int, list[list[int]]]):
                idx, b = args

                res = await self.client.embeddings.create(input=b, model=self.model)
                pbar.update(len(b))

                results[idx] = [np.array(embed.embedding) for embed in res.data]

            await spread_gather(task, enumerate(batches), n=5)  # 5 concurrent HTTP connections

        return [item for i in range(len(results)) for item in results[i]]


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
    except KeyError as e:
        raise NotImplementedError(name) from e


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
