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
            "You will be given one recipe's ingredients and instructions. "
            "Your task is to rate the summary on one criteria. "
            "Please make sure you read and understand the following instructions carefully. Please keep the"
            "recipe open while reviewing, and refer to it as needed.\n"
            "Evaluation Criteria:"
            "Coherence (1-10) - The collective quality of all sentences. Recipe coherence refers to the logical sense "
            "and clarity of a generated recipe. It assesses whether the recipe makes sense. It also evaluates the "
            "grammatical quality and simplicity of the "
            "recipe's language, ensuring that it is well-written and easy to understand. The recipe should not just "
            "be an ambiguous heap of related ingredients and steps, "
            "but should build step by step using the ingredients to give a coherent recipe."
            "Evaluation Steps:"
            "1. Read the recipe carefully and identify the ingredients and steps."
            "2. Read the recipe and evaluate its coherence. Check if the recipe contains any non-sense or gibberish"
            ", and if it is presented in a clear and logical order."
            "3. Assign a score for coherence on a scale of 1 to 10, where 1 is the lowest and 10 is the highest, and "
            "follow up with justifications for your evaluation"
            "Note: Coherence should not be confused with consistency, relevance and structural correctness"
            " of ingredients and instructions. Do not be concerned with any mistake that isn't directly related to "
            "coherence."),
        "human": "Evaluate this recipe :\n{recipe}\nScore: ",
    },
    "ingredient_relevance": {
        "system": (
            "You will be given one recipe's ingredients and instructions. "
            "Your task is to rate the summary on one criteria. "
            "Please make sure you read and understand the following instructions carefully. Please keep the"
            "recipe open while reviewing, and refer to it as needed. \n"
            "Evaluation Criteria:"
            "Relevance (1-10) - Recipe relevance refers to the extent to which the ingredients and steps included in "
            "the recipe align with the title of the intended dish, and its culinary restrictions (spicy, vegan, "
            "no-gluten, no bake, etc...) if and only if it is specified in the title. It assesses whether all the "
            "ingredients are relevant "
            "to the recipe and if all the steps contribute to achieving the desired outcome stated in the title. "
            "There should be no ingredients included that are unrelated or unnecessary for the specific recipe or "
            "that violate the culinary description. "
            "There should be no superfluous or unrelated steps that do not contribute to the intended dish.\n"
            "Evaluation Steps:"
            "1. Read the recipe carefully and identify the ingredients and steps.\n"
            "2. Read the recipe and evaluate its relevance. Check if the recipe contains any irrelevant ingredients or"
            "instructions that don't contribute to achieving the dish in the title. Make sure the recipe doesn't "
            "break the culinary restrictions of the title if and only if there is a restriction in the title.\n"
            "3. Assign a score for relevance on a scale of 1 to 10, where 1 is the lowest and 10 is the highest, and "
            "follow up with justifications for your evaluation"
            "IMPORTANT NOTE: Relevance should NOT be associated with consistency, coherence and structural correctness"
            "of ingredients and instructions. DO NOT be concerned with any mistake that isn't directly related to "
            "relevance."
        ),
        "human": "Evaluate this recipe :\n{recipe}\nScore: ",
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
    recipe_ingredients, recipe_instructions = parse_recipe(recipe)
    gold_ingredients, gold_instructions = parse_recipe(gold)
    results = rouge_metric.compute(
        predictions=["\n".join(recipe_ingredients) + "\n" + "\n".join(recipe_instructions)],
        references=["\n".join(gold_ingredients) + "\n" + "\n".join(gold_instructions)],
    )
    return round(results["rougeL"], 3)


def bleu(recipe: str, gold: str) -> float:
    """Metric Based Evaluation
    Calculates the BLEU score"""
    recipe_ingredients, recipe_instructions = parse_recipe(recipe)
    gold_ingredients, gold_instructions = parse_recipe(gold)
    results = bleu_metric.compute(
        predictions=["\n".join(recipe_ingredients) + "\n" + "\n".join(recipe_instructions)],
        references=["\n".join(gold_ingredients) + "\n" + "\n".join(gold_instructions)],
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


def linguistic_correctness(recipe: str) -> int:
    """Metric Based Evaluation
    Counts grammar and spelling mistakes detected using LanguageCheck"""

    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(recipe)
    tool.close()
    filtered_matches = [
        match
        for match in matches
        if match.ruleId not in ["UPPERCASE_SENTENCE_START", "WHITESPACE_RULE"]
    ]
    return len(filtered_matches)


async def ingredient_comparison(
    recipe_ingredients: str, gold_ingredients: str, logger: logging.Logger
) -> float:
    """LLM Based Evaluation
    How well the ingredients from the recipe match the ingredients from the gold recipe according to
    an LLM
    """
    r_start, r_end = recipe.index("Ingredients:\n") + len("Ingredients:\n"), recipe.index(
        "\nInstructions:"
    )
    g_start, g_end = gold.index("Ingredients:\n") + len("Ingredients:\n"), gold.index(
        "\nInstructions:"
    )
    r_ingredients = recipe[r_start:r_end]
    g_ingredients = gold[g_start:g_end]

    messages = format_message_history(
        "ingredient_comparison",
        recipeIngredients=r_ingredients,
        goldIngredients=g_ingredients,
    )

    resp = await chatgpt.agenerate(messages=[messages])
    log_text = (
        "ingredient_comparison prompt:\n"
        + textwrap.indent("\n".join(msg.content for msg in messages), "  ")
        + f"\ningredient_comparison response: {resp.generations[0][0].text} "
    )

    matches = int(resp.generations[0][0].text.split()[2])
    num_r_ingredients = len(r_ingredients.split("\n"))
    num_g_ingredients = len(g_ingredients.split("\n"))

    out = round((matches / num_r_ingredients + matches / num_g_ingredients) / 2, 3)
    logger.debug(f"ingredient_comparison result: {out}")
    response = resp.generations[0][0].text
    try:
        matches = int(response.split()[2])
        out = round((matches / len(recipe_ingredients) + matches / len(gold_ingredients)) / 2, 3)
        logger.debug(log_text + f"\ningredient_comparison result: {out}\n\n\n")
    except ValueError:
        out = 0
        logger.debug(f"FAIL- {log_text}\n\n\n")
    return out


async def ingredient_consistency(recipe: str, logger: logging.Logger) -> int:
    """Model Based Evaluation
    Does the ingredients' list accurately reflect the exact ingredients and amounts used in
    the directions according to an LLM?"""
    messages = format_message_history("ingredient_consistency", recipe=recipe)

    resp = await chatgpt.agenerate(messages=[messages])
    log_text = (
        "ingredient_consistency prompt:\n"
        + textwrap.indent("\n".join(msg.content for msg in messages), "  ")
        + f"\ningredient_consistency response: {resp.generations[0][0].text}"
    )

    response = resp.generations[0][0].text
    pattern = r"\d+"
    match = re.search(pattern, response)

    try:
        out = int(match.group())
        logger.debug(f"{log_text}\n ingredient_consistency result: {out}\n\n\n")
    except ValueError:
        if response.split()[2] == "no":
            logger.debug(f"{log_text}\n ingredient_consistency result: no\n\n\n")
            return 0
        out = 42
        logger.debug(f"FAIL- {log_text}\n\n\n")
    return out


async def step_order(recipe: str, logger: logging.Logger) -> bool:
    """Model Based Evaluation
    Does the order of the steps make sense according to an LLM?"""
    messages = format_message_history("step_order", recipe=recipe)

    resp = await chatgpt.agenerate(messages=[messages])
    log_text = (
        "step_order prompt:\n"
        + textwrap.indent("\n".join(msg.content for msg in messages), "  ")
        + f"\nstep_order response: {resp.generations[0][0].text}"
    )

    response = resp.generations[0][0].text
    if response.lower().startswith("true"):
        out = True
        logger.debug(f"{log_text}\n ingredient_consistency result: {out}\n\n\n")
    elif response.lower().startswith("false"):
        out = False
        logger.debug(f"{log_text}\n ingredient_consistency result: {out}\n\n\n")
    else:
        out = False
        logger.debug(f"FAIL- {log_text}\n\n\n")
    return out


async def coherence(recipe: str, logger: logging.Logger) -> int:
    """Model Based Evaluation
    Is the recipe clear and readable according to an LLM?"""
    messages = format_message_history("coherence", recipe=recipe)

    resp = await chatgpt.agenerate(messages=[messages])
    log_text = (
        "coherence prompt:\n"
        + textwrap.indent("\n".join(msg.content for msg in messages), "  ")
        + f"\ncoherence response: {resp.generations[0][0].text}"
    )

    response = resp.generations[0][0].text
    try:
        out = int(response.split("/")[0])
        logger.debug(f"{log_text}\n coherence result: {out}\n\n\n")
    except ValueError:
        out = 0
        logger.debug(f"FAIL- {log_text}\n\n\n")
    return out


async def ingredient_relevance(recipe: str, logger: logging.Logger) -> bool:
    """Model Based Evaluation
    Does the list of ingredients align with the culinary expectations of the recipe (e.g. no-bake,
    gluten-free...)?
    """
    messages = format_message_history("ingredient_relevance", recipe=recipe)

    resp = await chatgpt.agenerate(messages=[messages])
    log_text = (
        "ingredient_relevance prompt:\n"
        + textwrap.indent("\n".join(msg.content for msg in messages), "  ")
        + f"\ningredient_relevance response: {resp.generations[0][0].text}"
    )

    response = resp.generations[0][0].text
    if response.lower().startswith("true"):
        out = True
        logger.debug(f"{log_text}\ningredient_consistency result: {out}\n\n\n")
    elif response.lower().startswith("false"):
        out = False
        logger.debug(f"{log_text}\ningredient_consistency result: {out}\n\n\n")
    else:
        out = False
        logger.debug(f"FAIL- {log_text}\n\n\n")
    return out


async def evaluation(recipe: str, gold: Dict[str, str], logger: logging.Logger) -> Dict[str, Any]:
    """Evaluates a generated recipe using all the above defined metrics"""
    title = gold["title"]
    gold_ingredients = "Ingredients:\n" + "\n".join(gold["ingredients"])
    gold_instructions = "Instructions:\n" + "\n".join(gold["directions"])
    ings, dirs = parse_recipe(recipe)
    recipe_ingredients = "Ingredients:\n" + "\n".join(ings)
    recipe_instructions = "Instructions:\n" + "\n".join(dirs)
    async_tasks = [
        ingredient_comparison(recipe_ingredients, gold_ingredients, logger),
        ingredient_consistency(recipe_ingredients + recipe_instructions, logger),
        ingredient_relevance(title + "\n" + recipe_ingredients, logger),
        step_order(recipe_instructions, logger),
        coherence(recipe_ingredients + "\n\n" + recipe_instructions, logger),
    ]
    resp = await asyncio.gather(*async_tasks)
    return {
        "rouge": rouge(
            recipe_ingredients + "\n" + recipe_instructions,
            gold_ingredients + "\n" + gold_instructions,
        ),
        "bleu": bleu(
            recipe_ingredients + "\n" + recipe_instructions,
            gold_ingredients + "\n" + gold_instructions,
        ),
        "cosine similarity": cosine_sim(
            recipe_ingredients + "\n" + recipe_instructions,
            gold_ingredients + "\n" + gold_instructions,
        ),
        "linguistic errors": linguistic_correctness(
            recipe_ingredients + "\n" + recipe_instructions
        ),
        "ingredient similarity ratio": resp[0],
        "ingredient inconsistencies": resp[1],
        "ingredient relevance": resp[2],
        "step order": resp[3],
        "coherence": resp[4],
    }
