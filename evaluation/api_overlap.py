from dataset import Procedure


def get_apis(response: str):
    token_list = response.split("`")
    start_idx = 1
    api_list = []
    while start_idx < len(token_list):
        api_list.append(token_list[start_idx])
        start_idx += 2

    return api_list


def check_api_overlap(gold: Procedure, generated: str):
    gold_apis = get_apis(str(gold.steps))
    generated_apis = get_apis(generated)

    num_present = 0
    for api in gold_apis:
        if api in generated_apis:
            num_present += 1

    return num_present / len(gold_apis)
