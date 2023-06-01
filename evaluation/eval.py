""" RECIPE EVALUATION"""
import ast
import re
import time
from typing import Any, Dict
import asyncio
import evaluate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import language_tool_python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

evaluation_messages = {
    "ingredient_comparison": [
        SystemMessage(
            content="You are a evaluating two ingredients lists to see how well the first lists' "
            "ingredients match the second. Count how many ingredients are used in "
            "both lists, the wording or measurements don't have to be exactly the same as long as"
            "the ingredient is the same material. "
            "For example: There are 5 matching ingredients between the two lists: eggs, flour,"
            "vanilla extract, milk, and butter.; There are 2 matching ingredients between the two"
            "lists: peanuts and jelly; There are 0 matching ingredients between the two lists"
        ),
        HumanMessage(
            content=(
                "Here are the two ingredients' lists : ---LIST 1 START---\n{recipeIngredients}\n"
                "---LIST 1 END---\n---LIST 2 START---\n{goldIngredients}\n---LIST 2 END---\n"
            )
        ),
    ],
    "ingredient_consistency": [
        SystemMessage(
            content=(
                "You are evaluating recipes to count the number of inconsistencies between the "
                "steps and the ingredient list. An inconsistency is using an ingredient in the "
                "steps that is not mentioned in the ingredients' list, or not using the same "
                "amount of an ingredient as mentioned in the ingredients' list, etc... Start your "
                "response by stating the number of inconsistencies that there are. For example: "
                "There are 0 inconsistencies; There is 1 inconsistency; There are 2 "
                "inconsistencies; There are 4 inconsistencies; etc..."
            )
        ),
        HumanMessage(content="Evaluate this recipe :\n{recipe}"),
    ],
    "step_order": [
        SystemMessage(
            content="You are a evaluating recipes to make sure that the steps are in the correct, "
            "cohesive order. The answer is True if the steps are in the correct order, and False "
            "if the steps are in incorrect order. Start your answer by stating your answer: True "
            "or False, followed by your justification. For example: True. The recipe's steps are "
            "in a logical order; False. Step 3 should come after Step 4 because...; etc..."
        ),
        HumanMessage(content="Evaluate this recipe :\n{recipe}"),
    ],
    "coherence": [
        SystemMessage(
            content="You are a evaluating recipes to make sure that the recipe is coherent, clear "
            "and readable. You will give the recipe a score out of 10. Start your output with the "
            "score, then follow up with your justifications. For example : 10/10. The recipe is "
            "perfectly coherent; 9/10. The recipe is well-structured but has a minor "
            "imperfection...; etc..."
        ),
        HumanMessage(content="Evaluate this recipe :\n{recipe}"),
    ],
    "ingredient_relevance": [
        SystemMessage(
            content="You are a evaluating recipes to make sure that the ingredients and steps are "
            "in alignment with the title. Do the ingredients and recipe produce the dish in the "
            "title and are culinary conditions respected. The answer is True if there is alignment "
            "and False if not, followed by your explanation. For example: False. The recipe title "
            "indicates that it is gluten-free but some ingredients have gluten; True. The "
            "ingredients align with the culinary expectations of the recipe; False. The recipe "
            "title indicates that it is no-bake but the steps include a baking step; etc..."
        ),
        HumanMessage(content="Evaluate this recipe :\n{recipe}"),
    ],
}
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
chatgpt = ChatOpenAI()


def format_recipe(recipe) -> str:
    """Formats a recipe taken directly from the RecipeNLG dataset into a string we can use for our
    evaluation."""

    title = recipe.loc["title"]
    steps = ast.literal_eval(recipe.loc["directions"])
    ingredients = ast.literal_eval(recipe.loc["ingredients"])
    if len(steps) == 1:
        steps = steps[0].split(". ")
        steps = [step.strip('."') + "." for step in steps]
    steps = [str(i + 1) + ". " + steps[i].strip(' "') for i in range(len(steps))]
    ingredients = ["- " + ingredient.strip(' "') for ingredient in ingredients]
    ingredients = "\n".join(ingredients)
    steps = "\n".join(steps)
    result = f"{title}\n\n" f"Ingredients:\n{ingredients}\n\n" f"Instructions:\n{steps}"
    return result


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


async def ingredient_comparison(recipe: str, gold: str) -> float:
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

    messages = evaluation_messages["ingredient_comparison"]
    messages[1].content = messages[1].content.format(
        recipeIngredients=recipeIngredients, goldIngredients=goldIngredients
    )
    resp = await chatgpt.agenerate(messages=[messages])
    matches = int(resp.generations[0][0].text.split()[2][:5])
    numRecipeIngredients = len(recipeIngredients.split("\n"))
    numGoldIngredients = len(goldIngredients.split("\n"))
    return round((matches / numRecipeIngredients + matches / numGoldIngredients) / 2, 3)


async def ingredient_consistency(recipe: str) -> int:
    """Model Based Evaluation
    Does the ingredients' list accurately reflect the exact ingredients and amounts used in
    the directions according to an LLM?"""

    messages = evaluation_messages["ingredient_consistency"]
    messages[1].content = messages[1].content.format(recipe=recipe)
    resp = await chatgpt.agenerate(messages=[messages])
    response = resp.generations[0][0].text
    pattern = r"\d+"
    match = re.search(pattern, response)

    out = match.group() if match else 0

    return int(out)


async def step_order(recipe: str) -> bool:
    """Model Based Evaluation
    Does the order of the steps make sense according to an LLM?"""

    messages = evaluation_messages["step_order"]
    messages[1].content = messages[1].content.format(recipe=recipe)
    resp = await chatgpt.agenerate(messages=[messages])
    answer = resp.generations[0][0].text.split()[0]

    return answer.lower().startswith("true")


async def coherence(recipe: str) -> int:
    """Model Based Evaluation
    Is the recipe clear and readable according to an LLM?"""

    messages = evaluation_messages["coherence"]
    messages[1].content = messages[1].content.format(recipe=recipe)
    resp = await chatgpt.agenerate(messages=[messages])
    answer = resp.generations[0][0].text.split()[0].strip(".")

    return int(answer.split("/")[0])


async def ingredient_relevance(recipe: str) -> bool:
    """Model Based Evaluation
    Does the list of ingredients align with the culinary expectations of the recipe (e.g. no-bake,
    gluten-free...)?
    """

    messages = evaluation_messages["ingredient_relevance"]
    messages[1].content = messages[1].content.format(recipe=recipe)
    resp = await chatgpt.agenerate(messages=[messages])
    answer = resp.generations[0][0].text.split()[0]

    return answer.lower().startswith("true")


async def evaluation(recipe: str, gold: str) -> Dict[str, Any]:
    """Evaluates a generated recipe using all the above defined metrics"""
    async_tasks = [
        ingredient_comparison(recipe, gold),
        ingredient_consistency(recipe),
        ingredient_relevance(recipe),
        step_order(recipe),
        coherence(recipe)
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
