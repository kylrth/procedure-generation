""" RECIPE EVALUATION"""
import asyncio
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Any

import evaluate
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from language_tool_python import LanguageTool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recipenlg import format_recipe, parse_recipe


with Path.open(Path("./evaluation/evaluation-prompts.json")) as file:
    evaluation_messages = json.load(file)


def format_message_history(key: str, **kwargs):
    messages = evaluation_messages[key]

    return [
        SystemMessage(content=messages["system"].format(**kwargs)),
        HumanMessage(content=messages["human"].format(**kwargs)),
    ]


def log_output(caller: str, messages: list[BaseMessage], resp):
    def _format(msg: BaseMessage) -> str:
        if "\n" not in msg.content:
            return msg.__class__.__name__ + "(" + msg.content + ")"
        return msg.__class__.__name__ + "(\n  " + msg.content.replace("\n", "\n  ") + "\n)"

    return (
        f"{caller} prompt:\n"
        + textwrap.indent("\n".join(_format(msg) for msg in messages), "  ")
        + f"\n{caller} response: {resp.generations[0][0].text}"
    )


rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
chatgpt = ChatOpenAI(temperature=0.3)


def rouge(recipe: str, gold: str) -> float:
    """Metric Based Evaluation
    Calculates the ROUGE score"""
    r_ingredients, r_instructions = parse_recipe(recipe)
    g_ingredients, g_instructions = parse_recipe(gold)
    results = rouge_metric.compute(
        predictions=["\n".join(r_ingredients) + "\n" + "\n".join(r_instructions)],
        references=["\n".join(g_ingredients) + "\n" + "\n".join(g_instructions)],
    )
    return round(results["rougeL"], 3)


def bleu(recipe: str, gold: str) -> float:
    """Metric Based Evaluation
    Calculates the BLEU score"""
    r_ingredients, r_instructions = parse_recipe(recipe)
    g_ingredients, g_instructions = parse_recipe(gold)
    results = bleu_metric.compute(
        predictions=["\n".join(r_ingredients) + "\n" + "\n".join(r_instructions)],
        references=["\n".join(g_ingredients) + "\n" + "\n".join(g_instructions)],
    )
    return round(results["bleu"], 3)


def cosine_sim(recipe: str, gold: str) -> float:
    """Metric Based Evaluation
    Calculates cosine similarity on TF-IDF representation to measure similarity between generated
    and gold recipe
    """

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([recipe, gold])
    return round(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0], 3)


def linguistic_correctness(lt: LanguageTool, recipe: str) -> int:
    """Metric Based Evaluation
    Counts grammar and spelling mistakes detected using LanguageCheck"""
    matches = lt.check(recipe)
    filtered_matches = [
        match
        for match in matches
        if match.ruleId not in ["UPPERCASE_SENTENCE_START", "WHITESPACE_RULE"]
    ]
    return len(filtered_matches)


async def hallucination() -> float:
    """LLM Based Evaluation
    Receives several samples of completions of a recipe.
    Hallucination refers to a phenomenon where a language model generates text that is factually
    incorrect or nonsensical, but appears coherent and plausible on the surface.
    Hallucination will be measured by comparing the different samples and evaluating their
    similarity on key points of the recipe.
    """
    # Implement this function
    return 0


async def consistency(recipe: str, logger: logging.Logger) -> int:
    """Model Based Evaluation Recipe
    Receives ingredients and instructions
    Consistency refers to the alignment between the ingredients listed,
    their respective measurements, and their usage in the recipe steps, ensuring coherence and
     logical progression.
    It also encompasses the assurance that the recipe yields the intended dish, without
    contradictory instructions or logical inconsistencies throughout."""
    messages = format_message_history("consistency", recipe=recipe)

    resp = await chatgpt.agenerate(messages=[messages])
    log_text = log_output("consistency", messages, resp)
    response = resp.generations[0][0].text
    pattern = r"\d+"
    match = re.search(pattern, response)

    if match is not None:
        out = int(match.group())
        logger.debug(f"{log_text}\n consistency result: {out}\n\n\n")
    else:
        out = 0
        logger.warning(f"FAIL- {log_text}\n\n\n")
    return out


async def structure(recipe: str, logger: logging.Logger) -> int:
    """Model Based Evaluation Receives ingredients and instructions Recipe structure correctness
    refers to the adherence to a standardized format, including an organized ingredients list and
    step-by-step instructions, facilitating clear understanding and easy execution. It ensures
    that all listed ingredients are actual ingredients, and the steps provided are actionable and
    sequentially structured, enhancing clarity and usability."""
    messages = format_message_history("structure", recipe=recipe)

    resp = await chatgpt.agenerate(messages=[messages])
    log_text = log_output("structure", messages, resp)

    response = resp.generations[0][0].text
    pattern = r"\d+"
    match = re.search(pattern, response)

    if match is not None:
        out = int(match.group())
        logger.debug(f"{log_text}\n consistency result: {out}\n\n\n")
    else:
        out = 0
        logger.warning(f"FAIL- {log_text}\n\n\n")
    return out


async def coherence(recipe: str, logger: logging.Logger) -> int:
    """Model Based Evaluation Receives title and ingredients and instructions Recipe coherence
    refers to the logical consistency and clarity of a recipe, including the sequential order of
    steps and absence of gibberish or nonsensical information. It also encompasses the
    grammatical correctness and simplicity of the recipe, ensuring that it is easily
    understandable and makes sense to the reader."""
    messages = format_message_history("coherence", recipe=recipe)

    resp = await chatgpt.agenerate(messages=[messages])
    log_text = log_output("coherence", messages, resp)

    response = resp.generations[0][0].text
    pattern = r"\d+"
    match = re.search(pattern, response)

    if match is not None:
        out = int(match.group())
        logger.debug(f"{log_text}\n consistency result: {out}\n\n\n")
    else:
        out = 0
        logger.warning(f"FAIL- {log_text}\n\n\n")
    return out


async def relevance(recipe: str, logger: logging.Logger) -> int:
    """Model Based Evaluation Receives title + ingredients + steps Recipe relevance refers to the
    appropriateness and alignment of the ingredients used in a recipe with its title, ensuring
    that all ingredients are relevant to the intended dish. It also involves ensuring that each
    step in the recipe contributes towards achieving the desired outcome mentioned in the recipe
    title, avoiding any unnecessary or unrelated instructions.
    """
    messages = format_message_history("relevance", recipe=recipe)

    resp = await chatgpt.agenerate(messages=[messages])
    log_text = log_output("relevance", messages, resp)

    response = resp.generations[0][0].text
    pattern = r"\d+"
    match = re.search(pattern, response)

    if match is not None:
        out = int(match.group())
        logger.debug(f"{log_text}\n consistency result: {out}\n\n\n")
    else:
        out = 0
        logger.warning(f"FAIL- {log_text}\n\n\n")
    return out


async def evaluation(
    recipe: str, gold: dict[str, Any], lt: LanguageTool, logger: logging.Logger
) -> dict[str, Any]:
    """Evaluates a generated recipe using all the above defined metrics"""
    title = gold["title"][0]
    gold = format_recipe(gold["ingredients"][0], gold["directions"][0])
    r_ingredients, r_instructions = parse_recipe(recipe)
    recipe = format_recipe(r_ingredients, r_instructions)

    # run synchronous evaluations
    results = {
        "rouge": rouge(recipe, gold),
        "bleu": bleu(recipe, gold),
        "cosine similarity": cosine_sim(recipe, gold),
    }

    # run asynchronous evaluations
    async_tasks = {
        "linguistic errors": asyncio.to_thread(linguistic_correctness, lt, recipe),
        "consistency": consistency(recipe, logger),
        "relevance": relevance(title + "\n" + recipe, logger),
        "structure": structure(recipe, logger),
        "coherence": coherence(title + "\n" + recipe, logger),
    }
    resp = await asyncio.gather(*async_tasks.values())

    # add async results to dict
    for name, result in zip(async_tasks.keys(), resp, strict=True):
        results[name] = result

    return results
