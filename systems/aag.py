import asyncio
import logging
from pathlib import Path

import weaviate
import weaviate.classes as wvc
from datasets import Dataset

from lcstep import Procedure
from utils import spread_gather

from . import utils
from .interface import Result, System
from .model import Model


with (Path(__file__).parent / "prompts/concept_skills.txt").open() as f:
    _concept_skill_instructions = f.read().strip()


async def _get_concept_skills(model: Model, doc: str) -> list[Procedure]:
    prompt = model.build_prompt(prompt=doc, context=_concept_skill_instructions)

    res = await model.agenerate(prompt)

    return [Procedure.from_text(skill.strip()) for skill in res[0].split("NEW PROCEDURE")]


async def build_concept_skills(model: Model, concept_docs: Dataset) -> list[Procedure]:
    res = await spread_gather(
        lambda doc: _get_concept_skills(model, doc), concept_docs.iter(1), 10, len(concept_docs)
    )

    # flatten
    return [skill for sublist in res for skill in sublist]


_skills_collection = "Skills"


def setup_skills(
    logger: logging.Logger, store: weaviate.WeaviateClient, skills: list[Procedure]
) -> weaviate.Collection:
    """Create the skill library with the provided skills as a start.

    Since this collection will be updated over time, a pre-existing collection by the same name will
    be deleted.
    """
    if store.collections.exists(_skills_collection):
        logger.info("destroying existing Weaviate collection for skills")
        store.collections.delete(_skills_collection)

    logger.info("creating new Weaviate collection for skills")
    out = store.collections.create(
        name=_skills_collection,
        description="Skills extracted from high-level documentation about LangChain.",
        vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(),
        properties=[
            wvc.Property(
                name="goal",
                data_type=wvc.DataType.TEXT,
                description="The goal achieved by this skill",
            ),
            wvc.Property(
                name="steps",
                data_type=wvc.DataType.TEXT_ARRAY,
                description="The procedure to accomplish the goal, expressed as "
                "step-by-step instructions",
            ),
            wvc.Property(
                name="sidenote",
                data_type=wvc.DataType.TEXT,
                description="Optional extra information related to this procedure",
            ),
        ],
    )

    logger.info(f"uploading {len(skills)} skills to Weaviate collection")
    utils.weaviate_insert(
        logger, out, [{"goal": p.goal, "skill": p.steps, "sidenote": p.side_note} for p in skills]
    )

    return out


_api_ref_collection = "APIRef"


def setup_api_ref(
    logger: logging.Logger, store: weaviate.WeaviateClient, docs: list[dict[str, str]]
) -> weaviate.Collection:
    """Create the vector store for the API reference docs.

    Expects an iterable of dicts with keys "api" and "documentation".
    """
    if store.collections.exists(_api_ref_collection):
        logger.info("reusing existing Weaviate collection for API ref")
        out = store.collections.get(_api_ref_collection)
    else:
        logger.info("creating new Weaviate collection for API ref")
        out = store.collections.create(
            name=_api_ref_collection,
            description="API reference documentation for public methods of the LangChain Python "
            "library.",
            vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(),
            properties=[
                wvc.Property(
                    name="api",
                    data_type=wvc.DataType.TEXT,
                    description="The full import path of a LangChain method or class",
                ),
                wvc.Property(
                    name="chunk",
                    data_type=wvc.DataType.INT,
                    description="Zero-indexed chunk number in the documentation text",
                    skip_vectorization=True,
                    vectorize_property_name=False,
                ),
                wvc.Property(
                    name="documentation",
                    data_type=wvc.DataType.TEXT,
                    description="Full Markdown documentation for the method or class",
                ),
            ],
        )

    if len(out) == 0:
        logger.info(f"uploading {len(docs)} API ref chunks to Weaviate collection")
        utils.weaviate_insert(logger, out, docs)
    else:
        logger.info(f"using old data, {len(out)} chunks")

    return out


class AAG(System):
    store: weaviate.WeaviateClient
    skills: weaviate.Collection
    api_ref: weaviate.Collection
    model: Model

    def __init__(
        self,
        model: Model,
        store: weaviate.WeaviateClient,
    ):
        """Create a new AAG system that maintains the skill library in the Weaviate instance.

        The model will only return one result when calling generate or agenerate.

        If the weaviate store already contains a collection named "Skills", that collection is
        assumed to contain the bootstrapped skills.
        """
        self.model = model
        self.store = store
        self.skills = store.collections.get(_skills_collection)
        self.api_ref = store.collections.get(_api_ref_collection)

    def generate(self, query: str, logger: logging.Logger | None = None) -> Result:
        return asyncio.run(self.agenerate(query, logger))

    async def agenerate(self, query: str, _: logging.Logger | None = None) -> Result:
        # search skill library
        skills = self.find_relevant_skills(query, 5)

        # use model to filter out irrelevant skills
        skills = [skills[i] for i in await self.ask_relevance(query, skills)]

        if len(skills) == 0:
            # if nothing was similar enough, search docs
            procedure = await self.skill_from_docs(query)
        elif len(skills) == 1:
            procedure = skills[0]
        else:
            # ask the model to synthesize the skills into a single procedure
            procedure = await self.merge_skills(query, skills)

        # iteratively improve
        while "TODO" in procedure:
            procedure = await self.refine(query, procedure)

        # review and compare with requirements
        procedure = await self.review(query, procedure)

        # store in skill library

        return [procedure]

    def find_relevant_skills(self, text: str, n: int) -> list[str]:
        """Retrieve skills that are most relevant to the given text."""
        res = self.skills.query.near_text(
            query=text,
            limit=n,
        )

        return [obj.properties["data"]["Get"] for obj in res.objects]

    def store_skill(self, skill: str):
        """Add a new skill to the skill library."""
        self.skills.data.insert({"skill": skill})

    async def ask_relevance(self, goal: str, skills: list[str]) -> list[int]:
        """Ask the model if any of these skills are relevant enough to the given goal to consider
        starting from for creating a procedure for the goal."""

    async def skill_from_docs(self, goal: str) -> str:
        """Search the docs for information related to the goal, and ask the model to create a
        candidate procedure based on that information."""

    async def merge_skills(self, goal: str, skills: list[str]) -> str:
        """Ask the model to use these skills to generate a single procedure for the goal."""

    async def refine(self, goal: str, procedure: str) -> str:
        """Ask the model to handle a single TODO in the candidate procedure.

        The model designs a retrieval query to search the skill library, API ref, or concept docs
        in order to improve or complete the given TODO. Then, given the retrieved information, the
        model modifies the procedure to resolve the TODO.
        """

    async def review(self, goal: str, procedure: str) -> str:
        """Give the model another chance to review the procedure to ensure it meets requirements."""
