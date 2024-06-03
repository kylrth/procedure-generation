from openai import OpenAI
from typing import Any
from sentence_transformers import SentenceTransformer
import numpy as np

class OpenAIEmbedder:
    model: str
    client: Any
    
    def __init__(self, model_name:str):
        self.model = model_name
        self.client = OpenAI()
    
    def embed(self, text: list[str])->list[np.ndarray]:
        output = self.client.embeddings.create(input = text, model=self.model).data
        embedding_list = []
        for o in len(output):
            embedding_list.append(o.embedding)
        
        return embedding_list


class HFEmbedder:
    model: Any
    
    def __init__(self, model_name:str):
        if model_name == "all-mpnet-base-v2":
            self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        else:
            raise NotImplementedError

    
    def embed(self, text:list[str])->list[np.ndarray]:
        return self.model.encode(text, show_progress_bar=True, convert_to_numpy=True)
        
    

embedder_dict={
    "hf": HFEmbedder,
    "openai":OpenAIEmbedder
}


def embedder_from_name(embedder_str):
    service, model = embedder_str.lower().split("-", 1)
    embedder = embedder_dict[service]
    return embedder(model)
