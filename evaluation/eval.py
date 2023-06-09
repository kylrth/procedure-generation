""" RECIPE EVALUATION"""
import asyncio
import logging
import re
import textwrap
from typing import Any

import evaluate
import language_tool_python
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recipenlg import parse_recipe, format_recipe

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
    "consistency": {
        "system": (
            "You will be given one recipe's ingredients and instructions.\n"
            "Your task is to rate the summary on one criteria.\n"
            "Please make sure you read and understand the following instructions carefully. Please keep the"
            "recipe open while reviewing, and refer to it as needed. \n"
            "Evaluation Criteria:\n"
            "Consistency: (1-10) - Recipe consistency assesses whether the ingredients list aligns with the items "
            "used in the instructions and their corresponding measurements. It also evaluates whether the "
            "instructions end up "
            "presenting the dish as a complete, ready-to-serve outcome. Additionally, recipe consistency "
            "examines whether there are any contradictions within the instructions and ensures that the "
            "recipe maintains logical consistency overall. "
            "There should be no ingredients mentioned in the ingredients but not in the instructions and vice-versa. "
            "There should be no abrupt stop to the instructions before finishing the recipe. "
            "There should be no conflicting instructions that could cause confusion or contradiction with previous "
            "instructions.\n"
            "Evaluation Steps:\n"
            "1. Read the recipe carefully and identify the ingredients and instructions.\n"
            "2. Read the recipe and evaluate its consistency. Check if the recipe contains any inconsistent ingredients"
            "or instructions that cause a contradiction with previous statements of the recipe. Make sure the recipe "
            "ends with a ready to serve dish, as a consistent recipe should.\n"
            "3. Assign a score for consistency on a scale of 1 to 10, where 1 is the lowest and 10 is the highest, and "
            "follow up with justifications for your evaluation.\n"
            "IMPORTANT note: Consistency should NOT be associated with relevance, coherence and structural correctness"
            "of ingredients and instructions. DO NOT be concerned with any mistake that isn't directly related to "
            "consistency."
        ),
        "human": "Evaluate this recipe :\n{recipe}\nScore: ",
    },
    "structure": {
        "system": (
            "You will be given one recipe's ingredients and instructions.\n"
            "Your task is to rate the summary on one criteria.\n"
            "Please make sure you read and understand the following instructions carefully. Please keep the"
            "recipe open while reviewing, and refer to it as needed. \n"
            "Evaluation Criteria:\n"
            "Structural correctness: (1-10) - assesses whether the recipe structure is the proper format. "
            "The recipe should include an ingredients list, where every element of this list should be"
            "a measurement and ingredient. The recipe should also include an instructions list, where every instruction"
            " is one single imperative sentence or two at most. "       
            "There should be no element in the ingredients list that isn't a measurement and ingredient pair. "
            "There should be no element in the instructions list that is several actions packed into one line, "
            "or an element that isn't an imperative sentence.\n"
            "Evaluation Steps:\n"
            "1. Read the recipe carefully and identify the ingredients and instructions.\n"
            "2. Read the recipe and evaluate its structure. Check if the recipe contains an ingredient list and"
            "an instructions list. Check that the elements in these two lists are well-structured."
            "Make sure that the ingredients in the ingredient list are actual ingredients (with their measurements"
            "provided), and the instructions in the instruction list are actual instructions.\n"
            "3. Assign a score for structure on a scale of 1 to 10, where 1 is the lowest and 10 is the highest, and "
            "follow up with justifications for your evaluation.\n"
            "IMPORTANT NOTE: Structural correctness should NOT be associated with relevance, coherence and consistency"
            "of ingredients and instructions. DO NOT be concerned with any mistake that isn't directly related to "
            "the structure of the recipe."
        ),
        "human": "Evaluate this recipe :\n{recipe}\nScore: ",
    },
    "coherence": {
        "system": (
            "You will be given one recipe's ingredients and instructions.\n"
            "Your task is to rate the summary on one criteria.\n"
            "Please make sure you read and understand the following instructions carefully. Please keep the"
            "recipe open while reviewing, and refer to it as needed.\n"
            "Evaluation Criteria:\n"
            "Coherence (1-10) - The collective quality of all sentences. Recipe coherence refers to the logical sense "
            "and clarity of a generated recipe. It assesses whether the recipe makes sense. It also evaluates the "
            "grammatical quality and simplicity of the "
            "recipe's language, ensuring that it is well-written and easy to understand. The recipe should not just "
            "be an ambiguous heap of related ingredients and steps, "
            "but should build step by step using the ingredients to give a coherent recipe."
            "Evaluation Steps:\n"
            "1. Read the recipe carefully and identify the ingredients and steps.\n"
            "2. Read the recipe and evaluate its coherence. Check if the recipe contains any non-sense or gibberish"
            ", and if it is presented in a clear and logical order.\n"
            "3. Assign a score for coherence on a scale of 1 to 10, where 1 is the lowest and 10 is the highest, and "
            "follow up with justifications for your evaluation\n"
            "IMPORTANT Note: Coherence should NOT be confused with consistency, relevance and structural correctness"
            " of ingredients and instructions. DO NOT be concerned with any mistake that isn't directly related to "
            "coherence."),
        "human": "Evaluate this recipe :\n{recipe}\nScore: ",
    },
    "relevance": {
        "system": (
            "You will be given one recipe's title, ingredients, and instructions.\n"
            "Your task is to rate the summary on one criteria.\n"
            "Please make sure you read and understand the following instructions carefully. Please keep the"
            "recipe open while reviewing, and refer to it as needed. \n"
            "Evaluation Criteria:\n"
            "Relevance (1-10) - Recipe relevance refers to the extent to which the ingredients and steps included in "
            "the recipe align with the title of the intended dish, and its culinary restrictions (spicy, vegan, "
            "no-gluten, no bake, etc...) if and only if it is specified in the title. It assesses whether all the "
            "ingredients are relevant "
            "to the recipe and if all the steps contribute to achieving the desired outcome stated in the title. "
            "There should be no ingredients included that are unrelated or unnecessary for the specific recipe or "
            "that violate the culinary description. "
            "There should be no superfluous or unrelated steps that do not contribute to the intended dish.\n"
            "Evaluation Steps:\n"
            "1. Read the recipe carefully and identify the ingredients and steps.\n"
            "2. Read the recipe and evaluate its relevance. Check if the recipe contains any irrelevant ingredients or"
            "instructions that don't contribute to achieving the dish in the title. Make sure the recipe doesn't "
            "break the culinary restrictions of the title if and only if there is a restriction in the title.\n"
            "3. Assign a score for relevance on a scale of 1 to 10, where 1 is the lowest and 10 is the highest, and "
            "follow up with justifications for your evaluation.\n"
            "IMPORTANT note: Relevance should NOT be associated with consistency, coherence and structural correctness"
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


def log_output(caller: str, messages, resp):
    return (
        f"{caller} prompt:\n"
        + textwrap.indent("\n".join(msg.content for msg in messages), "  ")
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
    r_ingredients: str, g_ingredients: str, logger: logging.Logger
) -> float:
    """LLM Based Evaluation
    How well the ingredients from the recipe match the ingredients from the gold recipe according to
    an LLM
    """
    r_start, r_end = r_ingredients.index("Ingredients:\n") + len(
        "Ingredients:\n"
    ), r_ingredients.index("\nInstructions:")
    g_start, g_end = g_ingredients.index("Ingredients:\n") + len(
        "Ingredients:\n"
    ), g_ingredients.index("\nInstructions:")
    r_ingredients = r_ingredients[r_start:r_end]
    g_ingredients = g_ingredients[g_start:g_end]

    messages = format_message_history(
        "ingredient_comparison",
        recipeIngredients=r_ingredients,
        goldIngredients=g_ingredients,
    )

    resp = await chatgpt.agenerate(messages=[messages])
    log_text = log_output("ingredient_comparison", messages, resp)

    response = resp.generations[0][0].text
    try:
        # As per the prompt, the response here should be : There are X matches.
        # matches will extract the value X.
        matches = int(response.split()[2])
        num_r_ingredients = len(r_ingredients.split("\n"))
        num_g_ingredients = len(g_ingredients.split("\n"))
        out = round((matches / num_r_ingredients + matches / num_g_ingredients) / 2, 3)
        logger.debug(log_text + f"\ningredient_comparison result: {out}\n\n\n")
    except ValueError:
        out = 0
        logger.warning(f"FAIL- {log_text}\n\n\n")
    return out


async def consistency(recipe: str, logger: logging.Logger) -> int:
    """Model Based Evaluation Recipe
    Receives ingredients and instructions
    Consistency refers to the alignment between the ingredients listed,
    their respective measurements, and their usage in the recipe steps, ensuring coherence and logical progression.
    It also encompasses the assurance that the recipe yields the intended dish, without contradictory instructions or
    logical inconsistencies throughout. """
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
    """Model Based Evaluation
    Receives ingredients and instructions
    Recipe structure correctness refers to the adherence to a standardized format, including an organized ingredients
    list and step-by-step instructions, facilitating clear understanding and easy execution. It ensures that all listed
    ingredients are actual ingredients, and the steps provided are actionable and sequentially structured, enhancing
    clarity and usability."""
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
    """Model Based Evaluation
    Receives title and ingredients and instructions
    Recipe coherence refers to the logical consistency and clarity of a recipe, including the sequential order of steps
    and absence of gibberish or nonsensical information. It also encompasses the grammatical correctness and simplicity
    of the recipe, ensuring that it is easily understandable and makes sense to the reader."""
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
    """Model Based Evaluation
    Receives title + ingredients + steps
    Recipe relevance refers to the appropriateness and alignment of the ingredients used in a recipe with its title,
    ensuring that all ingredients are relevant to the intended dish. It also involves ensuring that each step in the
    recipe contributes towards achieving the desired outcome mentioned in the recipe title, avoiding any unnecessary or
    unrelated instructions.
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


async def evaluation(recipe: str, gold: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """Evaluates a generated recipe using all the above defined metrics"""
    title = gold["title"][0]
    full_gold = format_recipe(gold['ingredients'][0], gold['directions'][0])
    #ings, dirs = parse_recipe(recipe)
    #full_recipe = format_recipe(ings, dirs)
    full_recipe = recipe
    async_tasks = [
        consistency(full_recipe, logger),
        relevance(title + '\n' + full_recipe, logger),
        structure(full_recipe, logger),
        coherence(title + '\n' + full_recipe, logger),
    ]
    resp = await asyncio.gather(*async_tasks)
    return {
        "rouge": rouge(
            full_recipe,
            full_gold,
        ),
        "bleu": bleu(
            full_recipe,
            full_gold,
        ),
        "cosine similarity": cosine_sim(
            full_recipe,
            full_gold,
        ),
        "linguistic errors": linguistic_correctness(
            full_recipe
        ),
        "ingredient inconsistencies": resp[0],
        "ingredient relevance": resp[1],
        "step order": resp[2],
        "coherence": resp[3],
    }
