import logging
import weaviate
import weaviate.classes.config as wc
from langchain.schema import BaseMessage
from dataset import Procedure, Doc
import itertools
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
from weaviate import classes as wvc
from store import Store
from dataset import Doc

class Doc_store(Store):
    _rag_cache: str = "./cache/vectors/{name}"

    def __init__(self, store: weaviate.WeaviateClient, name: str, desc: str, cache_path: str, embedder: str):
        _rag_cache = _rag_cache.format(name=cache_path)
        super.__init__(store, name, desc, embedder)

    def setup_store(self, logger: logging.Logger) -> weaviate.collections.Collection:
        
        """Create a generic vector store for RAG."""
        if self.store.collections.exists(self.name):
            logger.info("reusing existing Weaviate collection")
            out = self.store.collections.get(self.name)
        else:
            logger.info("creating new Weaviate collection")
            out = self.store.collections.create(
                name=self.name,
                description=self.desc,
                properties=[
                    wc.Property(
                        name="title",
                        data_type=wc.DataType.TEXT,
                        description="The title of the document",
                    ),
                    wc.Property(
                        name="chunk",
                        data_type=wc.DataType.INT,
                        description="Zero-indexed chunk number in the document",
                        skip_vectorization=True,
                        vectorize_property_name=False,
                    ),
                    wc.Property(
                        name="contents",
                        data_type=wc.DataType.TEXT,
                        description="The contents of (this chunk of) the document",
                    ),
                ],
                vectorizer_config=wc.Configure.Vectorizer.none(),
            )
        return out        
        
    def populate(self, logger: logging.Logger, docs: list[Doc])->weaviate.collections.Collection:
        """Chunk, vectorize, cache, and upload the docs.

        Each doc dict must have keys "title" and "contents".
        """
        colln = self.setup_store(logger)
        
        rag_cache = Path(self._rag_cache)

        if not rag_cache.exists():
            logger.debug("chunking %d documents", len(docs))
            chunks = self.chunk_docs(docs, chunk_size=3000)

            logger.debug("vectorizing %d chunks", len(chunks))
            vectors = self.get_embeds(chunks, progress=True)

            # cache prepared data and vectors
            rag_cache.mkdir(parents=True, exist_ok=True)
            with (rag_cache / "prepared.json").open("w") as f:
                json.dump(chunks, f)
            np.save(rag_cache / "vectors.npy", vectors)
        else:
            logger.debug("loading cached chunks and vectors")
            # load cached data and vectors
            with (rag_cache / "prepared.json").open("r") as f:
                chunks = json.load(f)
            vectors = np.load(rag_cache / "vectors.npy")
        
        self.weaviate_insert(logger, colln, chunks, vectors)    
        return colln
    
    def chunk_docs(self,
        data: list[Doc], chunk_size: int = 4000, overlap: int = 200
    ) -> list[dict[str, str | int]]:
        """Takes a list of procedure (each should have keys "input", "output" and "steps"), and returns a chunked
        version.

        The output dicts have keys "title", "chunk" (0-indexed), and "contents".
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

        chunks = []
        for doc in data:
            # doc = proc.to_doc()
            split_doc = text_splitter.split_text(doc.contents)
            for idx, text_content in enumerate(split_doc):
                chunks.append(
                    {
                        "title": doc.title,
                        "chunk": idx,
                        "contents": text_content,
                    }
                )

        return chunks

    def get_docs(self, query: str) -> list[Doc]:
        """Returns the docs that will be inserted into the prompt."""
        embedded_query = self.embedder.embed([query])[0]
        docs = self.store.collections.get(self.name)

        res = docs.query.near_vector(
            near_vector=embedded_query.tolist(),
            limit=self.k,
            return_properties=["title", "contents"],
        )

        out = []
        for obj in res.objects:
            out.append(Doc(obj.properties["title"], obj.properties["contents"]))

        return out

    def format_str(self, text:dict[str, str|int]) -> str:
        title = text["title"]
        contents = text["contents"]
        contents = contents.replace("\n", "; ")
        formatted_str = f"Title: {title}, Contents: [{contents}]"
        return formatted_str
        
# _embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# def get_embeds(queries: list[str], progress: bool = False) -> list[np.ndarray]:
#     return _embedder.encode(queries, show_progress_bar=progress, convert_to_numpy=True)