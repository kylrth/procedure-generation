""" RECIPE EVALUATION"""
import asyncio
import logging
import re
import textwrap
from typing import Any, Dict, List

import evaluate
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
import language_tool_python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

evaluation_messages = {
    "ingredient_comparison": {
        "system": (
            "You are a evaluating two ingredients lists to see how well the first lists' "
            "ingredients match the second. Count how many ingredients are used in "
            "both lists, the wording or measurements don't have to be exactly the same as long as "
            "the ingredient is the same material. "
            "For example: There are 5 matching ingredients between the two lists: eggs, flour, "
            "vanilla extract, milk, and butter.; There are 2 matching ingredients between the two "
            "lists: peanuts and jelly; There are 0 matching ingredients between the two lists"
        ),
        "human": (
            "Here are the two ingredients' lists : ---LIST 1 START---\n{recipeIngredients}\n"
            "---LIST 1 END---\n---LIST 2 START---\n{goldIngredients}\n---LIST 2 END---\n"
        ),
    },
    "ingredient_consistency": {
        "system": (
            "You are evaluating recipes to count the number of inconsistencies between the steps "
            "and the ingredient list. An inconsistency is using an ingredient in the steps that is "
            "not mentioned in the ingredients' list, or not using the same amount of an ingredient "
            "as mentioned in the ingredients' list, etc... Start your response by stating the "
            "number of inconsistencies that there are. For example: There are 0 inconsistencies; "
            "There is 1 inconsistency; There are 2 inconsistencies; There are 4 inconsistencies; "
            "etc..."
        ),
        "human": "Evaluate this recipe :\n{recipe}",
    },
    "step_order": {
        "system": (
            "You are a evaluating recipes to make sure that the steps are in the correct, cohesive "
            "order. The answer is True if the steps are in the correct order, and False if the "
            "steps are in incorrect order. Start your answer by stating your answer: True or "
            "False, followed by your justification. For example: True. The recipe's steps are in a "
            "logical order; False. Step 3 should come after Step 4 because...; etc..."
        ),
        "human": "Evaluate this recipe :\n{recipe}",
    },
    "coherence": {
        "system": (
            "You are a evaluating recipes to make sure that the recipe is coherent, clear and "
            "readable. You will give the recipe a score out of 10. Start your output with the "
            "score, then follow up with your justifications. For example : 10/10. The recipe is "
            "perfectly coherent; 9/10. The recipe is well-structured but has a minor "
            "imperfection...; etc..."
        ),
        "human": "Evaluate this recipe :\n{recipe}",
    },
    "ingredient_relevance": {
        "system": (
            "You are a evaluating recipes to make sure that the ingredients and steps are in "
            "alignment with the title. Do the ingredients and recipe produce the dish in the title "
            "and are culinary conditions respected. The answer is True if there is alignment and "
            "False if not, followed by your explanation. For example: False. The recipe title "
            "indicates that it is gluten-free but some ingredients have gluten; True. The "
            "ingredients align with the culinary expectations of the recipe; False. The recipe "
            "title indicates that it is no-bake but the steps include a baking step; etc..."
        ),
        "human": "Evaluate this recipe :\n{recipe}",
    },
}


def format_message_history(key: str, **kwargs):
    messages = evaluation_messages[key]

    return [
        SystemMessage(content=messages["system"].format(**kwargs)),
        HumanMessage(content=messages["human"].format(**kwargs)),
    ]


rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
chatgpt = ChatOpenAI()


def rouge(recipe: str, gold: str) -> float:
    """Metric Based Evaluation
    Calculates the ROUGE score"""

    results = rouge_metric.compute(predictions=[recipe], references=[gold])
    return round(results["rougeL"], 3)


def bleu(recipe: str, gold: str) -> float:
    """Metric Based Evaluation
    Calculates the BLEU score"""

    results = bleu_metric.compute(predictions=[recipe], references=[gold])
    return round(results["bleu"], 3)


def cosine_sim(recipe: str, gold: str) -> float:
    """Metric Based Evaluation
    Calculates cosine similarity on TF-IDF representation to measure similarity between generated
    and gold recipe
    """

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([recipe, gold])
    return round(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0], 3)


def linguistic_correctness(recipe: str) -> int:
    """Metric Based Evaluation
    Counts grammar and spelling mistakes detected using LanguageCheck"""

    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(recipe)
    tool.close()
    filteredMatches = [
        match
        for match in matches
        if match.ruleId not in ["UPPERCASE_SENTENCE_START", "WHITESPACE_RULE"]
    ]
    return len(filteredMatches)


def log_messages(logger: logging.Logger, prefix: str, messages: List[BaseMessage]):
    text = textwrap.indent("\n".join(repr(msg) for msg in messages), "  ")
    logger.debug(prefix + text)


async def ingredient_comparison(recipe: str, gold: str, logger: logging.Logger) -> float:
    """LLM Based Evaluation
    How well the ingredients from the recipe match the ingredients from the gold recipe according to
    an LLM
    """
    recipeStartIdx, recipeEndIdx = recipe.index("Ingredients:\n") + len(
        "Ingredients:\n"
    ), recipe.index("\nInstructions:")
    goldStartIdx, goldEndIdx = gold.index("Ingredients:\n") + len("Ingredients:\n"), gold.index(
        "\nInstructions:"
    )
    recipeIngredients = recipe[recipeStartIdx:recipeEndIdx]
    goldIngredients = gold[goldStartIdx:goldEndIdx]

    messages = format_message_history(
        "ingredient_comparison",
        recipeIngredients=recipeIngredients,
        goldIngredients=goldIngredients,
    )

    resp = await chatgpt.agenerate(messages=[messages])
    log_messages(logger, "ingredient_comparison prompt:\n", messages)
    logger.debug(f"ingredient_comparison response: {resp.generations[0][0].text}")

    matches = int(resp.generations[0][0].text.split()[2])
    numRecipeIngredients = len(recipeIngredients.split("\n"))
    numGoldIngredients = len(goldIngredients.split("\n"))

    out = round((matches / numRecipeIngredients + matches / numGoldIngredients) / 2, 3)
    logger.debug(f"ingredient_comparison result: {out}")
    return out


async def ingredient_consistency(recipe: str, logger: logging.Logger) -> int:
    """Model Based Evaluation
    Does the ingredients' list accurately reflect the exact ingredients and amounts used in
    the directions according to an LLM?"""
    messages = format_message_history("ingredient_consistency", recipe=recipe)

    resp = await chatgpt.agenerate(messages=[messages])
    log_messages(logger, "ingredient_consistency prompt:\n", messages)
    logger.debug(f"ingredient_consistency response: {resp.generations[0][0].text}")

    response = resp.generations[0][0].text
    pattern = r"\d+"
    match = re.search(pattern, response)

    out = int(match.group() if match else 0)
    logger.debug(f"ingredient_consistency result: {out}")
    return out


async def step_order(recipe: str, logger: logging.Logger) -> bool:
    """Model Based Evaluation
    Does the order of the steps make sense according to an LLM?"""
    messages = format_message_history("step_order", recipe=recipe)

    resp = await chatgpt.agenerate(messages=[messages])
    log_messages(logger, "step_order prompt:\n", messages)
    logger.debug(f"step_order response: {resp.generations[0][0].text}")

    answer = resp.generations[0][0].text.split()[0]
    out = answer.lower().startswith("true")

    logger.debug(f"step_order result: {out}")
    return out


async def coherence(recipe: str, logger: logging.Logger) -> int:
    """Model Based Evaluation
    Is the recipe clear and readable according to an LLM?"""
    messages = format_message_history("coherence", recipe=recipe)

    resp = await chatgpt.agenerate(messages=[messages])
    log_messages(logger, "coherence prompt:\n", messages)
    logger.debug(f"coherence response: {resp.generations[0][0].text}")

    answer = resp.generations[0][0].text.split()[0].strip(".")
    out = int(answer.split("/")[0])

    logger.debug(f"coherence result: {out}")
    return out


async def ingredient_relevance(recipe: str, logger: logging.Logger) -> bool:
    """Model Based Evaluation
    Does the list of ingredients align with the culinary expectations of the recipe (e.g. no-bake,
    gluten-free...)?
    """
    messages = format_message_history("ingredient_relevance", recipe=recipe)

    resp = await chatgpt.agenerate(messages=[messages])
    log_messages(logger, "ingredient_relevance prompt:\n", messages)
    logger.debug(f"ingredient_relevance response: {resp.generations[0][0].text}")

    answer = resp.generations[0][0].text.split()[0]
    out = answer.lower().startswith("true")

    logger.debug(f"ingredient_relevance result: {out}")
    return out


async def evaluation(recipe: str, gold: str, logger: logging.Logger) -> Dict[str, Any]:
    """Evaluates a generated recipe using all the above defined metrics"""
    async_tasks = [
        ingredient_comparison(recipe, gold, logger),
        ingredient_consistency(recipe, logger),
        ingredient_relevance(recipe, logger),
        step_order(recipe, logger),
        coherence(recipe, logger),
    ]
    resp = await asyncio.gather(*async_tasks)
    return {
        "rouge": rouge(recipe, gold),
        "bleu": bleu(recipe, gold),
        "cosine similarity": cosine_sim(recipe, gold),
        "linguistic errors": linguistic_correctness(recipe),
        "ingredient similarity ratio": resp[0],
        "ingredient inconsistencies": resp[1],
        "ingredient relevance": resp[2],
        "step order": resp[3],
        "coherence": resp[4],
    }
