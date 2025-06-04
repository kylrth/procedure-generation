import json
import logging
import random
from typing import Awaitable, Callable, Type

from graph import Graph, Node
from model import Model

from .base import GraphProcedure as GProcedure
from .base import LinearProcedure, Step


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


async def get_node_list(sys_prompt: str, hum_prompt: str, model: Model, seed: int | None = None):
    out = model.build_prompt(hum_prompt, sys_prompt)
    if seed is not None:
        node_completion = await model.generate(out, seed=seed)
    else:
        node_completion = await model.generate(out)

    try:
        nodes = json.loads(node_completion)["Nodes"]
    except Exception as e:  # make sure any exception type here is caught by the outer except
        raise ValueError from e
    return nodes


async def refine_node(sys_prompt: str, hum_prompt: str, model: Model, seed: int | None = None):
    out = model.build_prompt(hum_prompt, sys_prompt)
    if seed is not None:
        node_completion = await model.generate(out, seed=seed)
    else:
        node_completion = await model.generate(out)
    try:
        node = json.loads(node_completion)
    except Exception as e:  # make sure any exception type here is caught by the outer except
        raise ValueError from e
    return node


async def get_edge_list(sys_prompt: str, hum_prompt: str, model: Model, seed: int | None = None):
    out = model.build_prompt(hum_prompt, sys_prompt)
    if seed is not None:
        edge_completion = await model.generate(out, seed=seed)
    else:
        edge_completion = await model.generate(out)
    try:
        nodes = json.loads(edge_completion)["Nodes"]
        edges = json.loads(edge_completion)["Edges"]
    except Exception as e:  # make sure any exception type here is caught by the outer except
        raise ValueError from e
    return nodes, edges


async def get_corrected_nodes_edges(
    sys_prompt: str, hum_prompt: str, model: Model, seed: int | None = None
):
    out = model.build_prompt(hum_prompt, sys_prompt)
    if seed is not None:
        edge_completion = await model.generate(out, seed=seed)
    else:
        edge_completion = await model.generate(out)
    try:
        nodes = json.loads(edge_completion)["Nodes"]
        edges = json.loads(edge_completion)["Edges"]
    except Exception as e:  # make sure any exception type here is caught by the outer except
        raise ValueError from e
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

        inp_out_list = edge["match"]
        assert len(inp_out_list) > 0  # noqa: S101  # AssertionError will be caught
        node_from.new_edge_to(node_to, inp_out_list[0])

        inp_list = node_inp_dict[edge["to"]]
        if inp_out_list[0] in inp_list:
            inp_list.remove(inp_out_list[0])
        if len(inp_out_list) == 2 and inp_out_list[1] in inp_list:
            inp_list.remove(inp_out_list[1])

        node_inp_dict[edge["to"]] = inp_list

    # Add input and output
    num_out = 0
    for name, inp_list in node_inp_dict.items():
        node_dict[name].add_inputs(*inp_list)

        if len(node_dict[name].outgoing) == 0:
            num_out += 1
            node_dict[name].add_outputs(proc_title)

    if num_out != 1:
        raise ValueError(f"{proc_title}: {num_out} outputs detected; should be 1")
    return GProcedure(*(node_dict.values()))


async def build_graph_from_linear_procedure(
    proc: LinearProcedure,
    model: Model,
    dataset: str,
    seed: int | None = None,
) -> GProcedure:
    steps = proc.steps
    ing = proc.input_ if dataset == "recipenlg" else None

    sys_prompt = instructions[dataset]
    hum_prompt_nodes = human_instruction_nodes[dataset].format(procedure_steps=steps, ing=ing)
    orig_nodes = await get_node_list(sys_prompt, hum_prompt_nodes, model, seed)

    for i, node in enumerate(orig_nodes):
        hum_prompt_node_refine = human_instruction_node_refine.format(node=node)
        orig_nodes[i] = await refine_node(sys_prompt, hum_prompt_node_refine, model, seed)

    hum_prompt_edges = human_instruction_edges.format(graph_nodes=orig_nodes)
    nodes, orig_edges = await get_edge_list(sys_prompt, hum_prompt_edges, model, seed)

    edges = filter_edges(nodes, orig_edges)

    if len(nodes) > 1:
        valid = False
        num_tries = 5
        while not valid and num_tries > 0:
            missed_nodes = check_node_edge_valid(nodes, edges)
            if len(missed_nodes) == 0:
                valid = True
                break

            hum_prompt_corr_edges = human_instruction_corr_edges.format(
                graph_nodes=nodes, edges=edges, missing_nodes=missed_nodes
            )
            nodes, edges = await get_corrected_nodes_edges(
                sys_prompt, hum_prompt_corr_edges, model, seed
            )

            seed = random.randint(0, 1000)
            num_tries -= 1

    proc_graph = make_graph_object(nodes, edges, proc.output)

    return proc_graph


async def create_simple_linear_graph(proc: LinearProcedure):
    node_list = []
    for i, step in enumerate(proc.steps):
        step_obj = Step(f"Step {i}", step)
        node_obj = Node(step_obj)
        if i > 0:
            # edge
            node_list[-1].new_edge_to(node_obj, "")
        node_list.append(node_obj)
    inp_list = [inp.strip() for inp in proc.input_.split(",")]
    node_list[0].add_inputs(*inp_list)
    node_list[-1].add_outputs(proc.output)
    return GProcedure(*node_list)


async def with_retries[
    T
](
    f: Callable[[int | None], Awaitable[T]], *, exc: tuple[Type[Exception], ...], retries: int = 5
) -> T:
    seed = None
    for _ in range(retries):
        try:
            return await f(seed)
        except exc:
            seed = random.randint(0, 1000)

    raise MaxRetriesError


class MaxRetriesError(Exception):
    def __init__(self):
        super().__init__("maximum retries reached")


async def build_graph_with_retries(
    logger: logging.Logger,
    proc: LinearProcedure,
    model: Model,
    dataset: str,
) -> GProcedure:
    """Build the graph and retry up to 4 times with different seeds if it fails. If it fails all
    those times, create a simple linear graph as a failure mode."""
    try:
        return await with_retries(
            lambda seed: build_graph_from_linear_procedure(proc, model, dataset, seed=seed),
            exc=(Graph.UnreachableError, ValueError, KeyError, AssertionError, TypeError),
        )
    except MaxRetriesError:
        logger.warning("max retries reached; building simple linear graph")
        return await create_simple_linear_graph(proc)
