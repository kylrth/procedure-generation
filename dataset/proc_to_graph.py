import json
import os
import pickle
import random

from dataset import LinearProcedure
from dataset.base import GraphProcedure as GProcedure
from dataset.base import Step
from graph import Graph, Node
from model import Model
from utils import log


random.seed(1)

instructions = {
    "recipenlg": (
        "You are an expert at recipes. Your goal is to break down the "
        "given recipe into atomic steps where each step performs "
        "a single action, just like a function in coding. The functions should be general "
        "cooking operations that can be used across multiple different recipes. "
        "Eventually, the task is to represent the recipe graphically with nodes as the functions "
        "and edges relating the dependencies between those functions."
    ),
    "lcstep": (
        "You are an expert at programming with LangChain library. Your "
        "goal is to break down the "
        "given procedure into atomic steps where each step performs "
        "a single action just like a function in coding. The functions should be general "
        "enough so that they can be used across multiple different procedures. "
        "Eventually, the task is to represent the procedure graphically with nodes as the "
        "functions and edges relating the dependencies between those functions."
    ),
    "champ": (
        "You are a maths expert. Your "
        "goal is to break down the given solution procedure for a question "
        "into atomic steps where each step performs "
        "a single action, just like a function in coding. The functions should be general "
        "enough so that they can be used across multiple different maths "
        "problems. "
        "Eventually, the task is to represent the solution graphically with nodes as the functions "
        "and edges relating the dependencies between those functions."
    ),
}

human_instruction_nodes = {
    "recipenlg": (
        "Please consider the procedure and the provided list of ingredients below:"
        "\n\n[BEGIN INGREDIENTS]\n{ing}\n[END INGREDIENTS]\n[BEGIN STEPS]\n{procedure_steps}\n"
        "[END STEPS]\n\nCarefully think about it step-by-step and identify the "
        "list of nodes that represent the full procedure with all the details preserved. Use the "
        "ingredients list to resolve any references to the ingredients in the provided steps. "
        "Your output should be a valid JSON object in the format below:\n"
        '{{\n"Analysis": "<thoughts about the steps",\n"Nodes": [<list of Node>]\n}}\n'
        "where each Node is another JSON object structured as:\n"
        '{{\n"name": "<name of the node>",\n"description": '
        '"<description of node borrowed from steps>",\n'
        '"inputs": [<list of input>],\n "output": "<output of the node>"\n}}\n'
        "and each input is a string. Remember that "
        "each node should perform a single action with the required inputs. "
        "Please always make sure to include specific details from "
        "the provided procedure like cooking time, temperature and full path "
        "to the APIs in your descriptions of the nodes."
    ),
    "lcstep": (
        "Please consider the procedure below:"
        "\n\n{procedure_steps}\n\nCarefully think about it step-by-step and identify the "
        "list of nodes that represent the full procedure with all the details preserved. "
        "Your output should be a valid JSON object in the format below:\n"
        '{{\n"Analysis": "<thoughts about the steps",\n"Nodes": [<list of Node>]\n}}\n'
        "where each Node is another JSON object structured as:\n"
        '{{\n"name": "<name of the node>",\n"description": '
        '"<description of node borrowed from steps>",\n'
        '"inputs": [<list of input>],\n "output": "<output of the node>"\n}}\n'
        "and each input is a string. Remember that "
        "each node should perform a single action with the required inputs. "
        "Please always make sure to include specific details from "
        "the provided procedure like cooking time, temperature and full path "
        "to the APIs in your descriptions of the nodes.{ing}"
    ),
}

human_instruction_node_refine = (
    "Consider the following node:\n"
    "\n\n{node}\n\nIf required, add missing inputs to the node according to the "
    "description. Your output should be a single valid JSON object structured as:\n"
    '{{\n"name": "<name of the node>",\n"description": '
    '"<description of node input in the prompt>",\n'
    '"inputs": [<list of input>],\n "output": "<output of the node>"\n}}\n'
    "and each input is a string."
)

human_instruction_edges = (
    "Please consider the list of identified graph nodes below:"
    "\n\n{graph_nodes}\n\nCarefully think about them step-by-step and identify the "
    "dependencies between the nodes. These dependencies will be represented by directed "
    "edges between the nodes. For every identified edge, find the input of the child that "
    "corresponds to the output of the parent. We call this [<parent output>, <child input>] "
    "list as match. "
    "Your output should be a valid JSON object in the format below:\n"
    '{{\n"Analysis": "<thoughts about the dependencies>",\n"Nodes": [<list of Node>]\n'
    '"Edges": [<list of Edge>]\n}}\n'
    "where each Edge is another JSON object structured as:\n"
    '{{\n"from": "<name of the from node>",\n"to": "<name of the to node>",\n"match": <match>\n}}\n'
    "where match is a list of strings defined above. "
    "Make sure to not modify the nodes at all in your response."
)

human_instruction_corr_edges = (
    "Please consider the list of identified graph nodes and edges below:"
    "\n\n[BEGIN NODES]\n{graph_nodes}\n[END NODES]\n\n[BEGIN EDGES]\n{edges}\n[END EDGES]\n\n"
    "In the above graph, following nodes have no edges to other nodes:\n\n{missing_nodes}\n\n"
    "Carefully think about them step-by-step and connect them "
    "to other nodes. These dependencies will be represented by directed "
    "edges between the nodes. For every identified edge, find the input of the child that "
    "corresponds to the output of the parent. We call this [<parent output>, <child input>] list "
    "as match. Your output should be a valid JSON object in the format below:\n"
    '{{\n"Analysis": "<thoughts about the new dependencies>",\n"Nodes": [<list of Node>]\n'
    '"Edges": [<list of Edge>]\n}}\n'
    "where each Edge is another JSON object structured as:\n"
    '{{\n"from": "<name of the from node>",\n"to": "<name of the to node>",\n"match": <match>\n}}\n'
    "where match is a list of strings defined above. "
    "Make sure to not modify the nodes at all in your response."
)


async def get_node_list(
    logger: log.InstanceLogger, sys_prompt: str, hum_prompt: str, model: Model, seed: int = None
):
    out = model.build_prompt(hum_prompt, sys_prompt)
    if seed is not None:
        node_completion = await model.generate(out, seed=seed)
    else:
        node_completion = await model.generate(out)

    try:
        nodes = json.loads(node_completion)["Nodes"]
    except:
        # print(node_completion)
        raise ValueError
    return nodes


async def refine_node(
    logger: log.InstanceLogger, sys_prompt: str, hum_prompt: str, model: Model, seed: int = None
):
    out = model.build_prompt(hum_prompt, sys_prompt)
    if seed is not None:
        node_completion = await model.generate(out, seed=seed)
    else:
        node_completion = await model.generate(out)
    try:
        node = json.loads(node_completion)
    except:
        # print(node_completion)
        raise ValueError
    return node


async def get_edge_list(
    logger: log.InstanceLogger, sys_prompt: str, hum_prompt: str, model: Model, seed: int = None
):
    out = model.build_prompt(hum_prompt, sys_prompt)
    if seed is not None:
        edge_completion = await model.generate(out, seed=seed)
    else:
        edge_completion = await model.generate(out)
    try:
        nodes = json.loads(edge_completion)["Nodes"]
        edges = json.loads(edge_completion)["Edges"]
        # output_node = json.loads(edge_completion)["Output Node"]
    except:
        # print(edge_completion)
        raise ValueError
    return nodes, edges


async def get_corrected_nodes_edges(
    logger: log.InstanceLogger, sys_prompt: str, hum_prompt: str, model: Model, seed: int = None
):
    out = model.build_prompt(hum_prompt, sys_prompt)
    if seed is not None:
        edge_completion = await model.generate(out, seed=seed)
    else:
        edge_completion = await model.generate(out)
    try:
        nodes = json.loads(edge_completion)["Nodes"]
        edges = json.loads(edge_completion)["Edges"]
    except:
        # print(edge_completion)
        raise ValueError
    return nodes, edges


def check_node_edge_valid(nodes, edges):
    missed_nodes = []
    for node in nodes:
        found = False
        for edge in edges:
            if edge["from"] == node["name"] or edge["to"] == node["name"]:
                found = True
                break
        if not found:
            missed_nodes.append(node["name"])
    return missed_nodes


def filter_edges(nodes, edges):
    node_name_list = [node["name"] for node in nodes]
    edge_list = []
    for edge in edges:
        if edge["from"] in node_name_list and edge["to"] in node_name_list:
            edge_list.append(edge)

    return edge_list


def make_graph_object(nodes, edges, proc_title):
    node_dict = {}
    node_inp_dict = {}
    for node in nodes:
        step_obj = Step(node["name"], node["description"])
        node_obj = Node(step_obj)
        node_dict[node["name"]] = node_obj
        node_inp_dict[node["name"]] = node["inputs"]

    for edge in edges:
        node_from = node_dict[edge["from"]]
        node_to = node_dict[edge["to"]]

        if node_from == node_to:
            continue
        else:
            inp_out_list = edge["match"]
            assert len(inp_out_list) > 0
            node_from.new_edge_to(node_to, inp_out_list[0])

            inp_list = node_inp_dict[edge["to"]]
            if inp_out_list[0] in inp_list:
                inp_list.remove(inp_out_list[0])
            if len(inp_out_list) == 2:
                if inp_out_list[1] in inp_list:
                    inp_list.remove(inp_out_list[1])

            node_inp_dict[edge["to"]] = inp_list

    # Add input and output
    for name, inp_list in node_inp_dict.items():
        node_dict[name].add_inputs(*inp_list)

        if len(node_dict[name].outgoing) == 0:
            node_dict[name].add_outputs(proc_title)

    return GProcedure(*(node_dict.values()))


async def get_graph_from_linear_procedure(
    logger: log.InstanceLogger,
    proc: LinearProcedure,
    model: Model,
    dataset: str,
    seed: int = None,
):
    steps = proc.steps
    ing = proc.input_ if dataset == "recipenlg" else None

    sys_prompt = instructions[dataset]
    hum_prompt_nodes = human_instruction_nodes[dataset].format(procedure_steps=steps, ing=ing)
    orig_nodes = await get_node_list(logger, sys_prompt, hum_prompt_nodes, model, seed)

    for i, node in enumerate(orig_nodes):
        hum_prompt_node_refine = human_instruction_node_refine.format(node=node)
        orig_nodes[i] = await refine_node(logger, sys_prompt, hum_prompt_node_refine, model, seed)

    hum_prompt_edges = human_instruction_edges.format(graph_nodes=orig_nodes)
    nodes, edges = await get_edge_list(logger, sys_prompt, hum_prompt_edges, model, seed)

    edges = filter_edges(nodes, edges)

    if len(nodes) > 1:
        valid = False
        num_tries = 5
        while not valid and num_tries > 0:
            missed_nodes = check_node_edge_valid(nodes, edges)
            if len(missed_nodes) == 0:
                valid = True
            else:
                print("going into correction mode")
                hum_prompt_corr_edges = human_instruction_corr_edges.format(
                    graph_nodes=nodes, edges=edges, missing_nodes=missed_nodes
                )
                nodes, edges = await get_corrected_nodes_edges(
                    logger, sys_prompt, hum_prompt_corr_edges, model, seed
                )
                seed = random.randint(0, 1000)
                num_tries -= 1
    proc_graph = make_graph_object(nodes, edges, proc.output)
    return proc_graph


async def create_simple_linear_graph(logger: log.InstanceLogger, proc: LinearProcedure):
    node_list = []
    for i, step in enumerate(proc.steps):
        step_obj = Step(f"Step {i}", step)
        node_obj = Node(step_obj)
        node_list.append(node_obj)
    inp_list = [inp.strip() for inp in proc.input_.split(",")]
    node_list[0].add_inputs(*inp_list)
    node_list[-1].add_outputs(proc.output)
    return GProcedure(*node_list)


async def create_graphs_for_graph_store(
    logger: log.InstanceLogger,
    proc_id: int,
    proc: LinearProcedure,
    model: Model,
    dataset: str,
    seed: int | None = None,
    save_pkl: bool = True,
):
    num_retries = 5
    while num_retries > 0:
        try:
            graph = await get_graph_from_linear_procedure(logger, proc, model, dataset, seed=seed)
            if save_pkl:
                os.makedirs(f"./dataset/graphs/{dataset}", exist_ok=True)
                with open(f"./dataset/graphs/{dataset}/{proc_id}.pkl", "wb") as f:
                    pickle.dump(graph, f)
                return
            else:
                return graph
        except (Graph.UnreachableError, ValueError, KeyError, AssertionError, TypeError):
            seed = random.randint(0, 1000)
            num_retries -= 1
    print(f"Skipping Procedure {proc_id}")
    if not save_pkl:
        graph = await create_simple_linear_graph(logger, proc)
        return graph
