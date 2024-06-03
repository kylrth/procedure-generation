import re

from langchain.schema import HumanMessage, SystemMessage

from dataset import Procedure
from systems import Model


def lcstep_all_resources_used(gold: Procedure, generated: str):
    if gold.input_ == "":
        return 1

    resource_str = gold.input_
    resource_str_list = resource_str.split(",")

    filtered_sentence = []
    for w in resource_str_list:
        w = w.lower()
        if w.startswith("an"):
            w = w[3:]
        elif w.startswith("a"):
            w = w[2:]
        elif w.startswith("the"):
            w = w[4:]

        filtered_sentence.append(w)

    resource_present = 0

    for resource_word in filtered_sentence:
        if resource_word in generated:
            resource_present += 1

    return resource_present / len(filtered_sentence)


def recipenlg_all_ingredients_used(model: Model, gold: Procedure, generated: str):
    ing_list = gold.input_
    prompt = [
        SystemMessage(
            content=(
                "Please rephrase the ingredient list below keeping only the name of the ingredient "
                "as it would appear in a recipe. For example, 8 oz. penne or other tubular pasta "
                "becomes 'pasta', 1 1/2 c. frozen green beans becomes 'green beans', and 2 Tbsp. "
                "chopped flat leaf parsley becomes 'parsley'."
            )
        ),
        HumanMessage(
            content="Referring to above examples, modify the list below:\n"
            + str(ing_list)
            + "\nYour response should be a python list starting with [ and end with ]. Remove the "
            "text enclosed in ()."
        ),
    ]

    trim_ing_list = model.generate(prompt)
    list_items = re.sub(r"[\s]", "", trim_ing_list)[1:-1].split(",")
    ing_found = 0
    for ing in list_items:
        if ing in generated:
            ing_found += 1

    return ing_found / len(list_items)
