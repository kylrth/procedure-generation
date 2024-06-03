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

class Procedure_store(Store):
    _aag_cache: str = "./cache/vectors/{name}"

    def __init__(self, store: weaviate.WeaviateClient, name: str, desc: str, cache_path: str, embedder:str):
        _aag_cache = _aag_cache.format(name=cache_path)
        super.__init__(store, name, desc, embedder)

    def setup_store(self, logger: logging.Logger) -> weaviate.collections.Collection:
        
        """Create the procedural memory with the provided procedures as the start."""
        if self.store.collections.exists(self.name):
            logger.info("destroying existing Weaviate collection for skills")
            self.store.collections.delete(self.name)

        logger.info("creating new Weaviate collection for skills")
        out = self.store.collections.create(
            name=self.name,
            description=self.desc,
            vectorizer_config=wc.Configure.Vectorizer.none(),
            properties=[
                wc.Property(
                    name="input",
                    data_type=wc.DataType.TEXT,
                    description="Resources for this procedure",
                ),
                wc.Property(
                    name="output",
                    data_type=wc.DataType.TEXT,
                    description="The goal achieved by this procedure",
                ),
                wc.Property(
                    name="steps",
                    data_type=wc.DataType.TEXT_ARRAY,
                    description="The procedure to accomplish the goal, expressed as "
                    "step-by-step instructions",
                ),
            ],
        )

        return out
    
    def get_properties_from_procedures(self, data: list[Procedure])->list[dict[str, Any]]:
        prop_list = []
        for proc in data:
            prop_list.append(proc.to_dict())
        
        return prop_list
    
    def populate(self, logger: logging.Logger, docs: list[Procedure])->weaviate.collections.Collection:
        """Chunk, vectorize, cache, and upload the docs.

        Each doc dict must have keys "title" and "contents".
        """
        colln = self.setup_store(logger)
        
        aag_cache = Path(self._aag_cache)

        if not aag_cache.exists():
            properties = self.get_properties_from_procedures(docs)
            logger.debug("vectorizing %d docs", len(docs))
            vectors = self.get_embeds(docs)

            # cache prepared data and vectors
            aag_cache.mkdir(parents=True, exist_ok=True)
            with (aag_cache / "prepared.json").open("w") as f:
                json.dump(chunks, f)
            np.save(aag_cache / "vectors.npy", vectors)
        else:
            logger.debug("loading cached data and vectors")
            # load cached data and vectors
            with (aag_cache / "prepared.json").open("r") as f:
                chunks = json.load(f)
            vectors = np.load(aag_cache / "vectors.npy")
        
        self.weaviate_insert(logger, colln, properties, vectors)    
        return colln

    def get_docs(self, query: str) -> list[Procedure]:
        """Returns the procedures that will be inserted into the prompt."""
        embedded_query = self.embedder.embed([query])[0]
        docs = self.store.collections.get(self.name)

        res = docs.query.near_vector(
            near_vector=embedded_query.tolist(),
            limit=self.k,
            return_properties=["input", "output", "steps"],
        )

        out = []
        for obj in res.objects:
            out.append(Procedure(obj.properties["input"], obj.properties["output"], obj.properties["steps"]))

        return out

    def format_str(self, text: Procedure) -> str:
        formatted_str = f"Input: {text._input}, "
        formatted_str += f"Output: {text.output}, "
        format_steps = '; '.join(text.steps)
        formatted_str += f"Steps: [{format_steps}]"
        return formatted_str