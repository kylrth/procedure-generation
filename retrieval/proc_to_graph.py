from .graph_procedure_store import Graph, Input, Output, Step, Node, Edge
from systems import Model
from utils import log
from dataset import Procedure
import json

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
        "Eventually, the task is to represent the procedure graphically with nodes as the functions "
        "and edges relating the dependencies between those functions."
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

human_instruction_nodes = (
    "Please consider the procedure below:"
    "\n\n{procedure_steps}\n\nCarefully think about it step-by-step and identify the "
    "list of nodes that represent the full procedure with all the details preserved. "
    "Your output should be a valid JSON object in the format below:\n"
    '{{\n"Analysis": "<thoughts about the steps",\n"Nodes": [<list of Node>]\n}}\n'
    "where each Node is another JSON object structured as:\n"
    '{{\n"name": "<name of the node>",\n"description": "<description of node borrowed from steps>",\n'
    '"inputs": [<list of input>],\n "output": "<output of the node>"\n}}\n'
    "and each input is a string. Remember that "
    "each node should perform a single action with the required inputs. "
    "Please always make sure to include specific details from "
    "the provided procedure like cooking time, temperature and full path "
    "to the APIs in your descriptions of the nodes."
)

human_instruction_node_refine = (
    "Consider the following node:\n"
    "\n\n{node}\n\nIf required, add missing inputs to the node according to the "
    "description. Your output should be a single valid JSON object structured as:\n"
    '{{\n"name": "<name of the node>",\n"description": "<description of node input in the prompt>",\n'
    '"inputs": [<list of input>],\n "output": "<output of the node>"\n}}\n'
    "and each input is a string."
)


human_instruction_edges = (
    "Please consider the list of identified graph nodes below:"
    "\n\n{graph_nodes}\n\nCarefully think about them step-by-step and identify the "
    "dependencies between the nodes. These dependencies will be represented by directed "
    "edges between the nodes. For every identified edge, match the text of the output of the parent "
    "to the corresponding input of the child."
    "Your output should be a valid JSON object in the format below:\n"
    '{{\n"Analysis": "<thoughts about the dependencies>",\n"Nodes": [<list of Node>]\n'
    '"Edges": [<list of Edge>]\n}}\n'
    "where each Edge is another JSON object structured as:\n"
    '{{\n"from": "<name of the from node>",\n"to": "<name of the to node>"\n}}\n'
    "The Nodes should contain the same set of nodes as given in the prompt but with inputs "
    "modified according to the instructions above."
)

human_instruction_corr_edges = (
    "Please consider the list of identified graph nodes and edges below:"
    "\n\n[BEGIN NODES]\n{graph_nodes}\n[END NODES]\n\n[BEGIN EDGES]\n{edges}\n[END EDGES]\n\n"
    "In the above graph, following nodes have no edges to other nodes:\n\n{missing_nodes}\n\n"
    "Carefully think about them step-by-step and connect them "
    "to other nodes. These dependencies will be represented by directed "
    "edges between the nodes. For every identified edge, match the text of the output of the parent "
    "to the corresponding input of the child while preserving other inputs in the list."
    "Your output should be a valid JSON object in the format below:\n"
    '{{\n"Analysis": "<thoughts about the new dependencies>",\n"Nodes": [<list of Node>]\n'
    '"Edges": [<list of Edge>]\n}}\n'
    "where each Edge is another JSON object structured as:\n"
    '{{\n"from": "<name of the from node>",\n"to": "<name of the to node>"\n}}\n'
    "The Nodes should contain the same set of nodes as given in the prompt but with inputs "
    "modified according to the instructions above."
)

# sys_instructions_conv_to_graph = (
#     "You are an expert on data modelling. Your goal is to convert the procedure steps "
#     "into a graphical representation, with one node for each step and edges relating the "
#     "nodes."
# )


async def get_node_list(logger: log.InstanceLogger, sys_prompt: str, hum_prompt: str, model: Model):
    out = model.build_prompt(hum_prompt, sys_prompt)
    node_completion = await model.generate(out)

    try:
        nodes = json.loads(node_completion)["Nodes"]
    except:
        raise ValueError
    return nodes


async def refine_node(logger: log.InstanceLogger, sys_prompt: str, hum_prompt: str, model: Model):
    out = model.build_prompt(hum_prompt, sys_prompt)
    node_completion = await model.generate(out)
    print(node_completion)
    try:
        node = json.loads(node_completion)
    except:
        raise ValueError
    return node


async def get_edge_list(logger: log.InstanceLogger, sys_prompt: str, hum_prompt: str, model: Model):
    out = model.build_prompt(hum_prompt, sys_prompt)
    edge_completion = await model.generate(out)
    try:
        nodes = json.loads(edge_completion)["Nodes"]
        edges = json.loads(edge_completion)["Edges"]
    except:
        print(edge_completion)
        raise ValueError
    return nodes, edges


async def get_corrected_nodes_edges(
    logger: log.InstanceLogger, sys_prompt: str, hum_prompt: str, model: Model
):
    out = model.build_prompt(hum_prompt, sys_prompt)
    edge_completion = await model.generate(out)
    try:
        nodes = json.loads(edge_completion)["Nodes"]
        edges = json.loads(edge_completion)["Edges"]
    except:
        print(edge_completion)
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


async def get_graph_from_linear_procedure(
    logger: log.InstanceLogger, steps: list[str], model: Model, dataset: str
) -> Graph[Step]:
    sys_prompt = instructions[dataset]
    hum_prompt_nodes = human_instruction_nodes.format(procedure_steps=steps)
    orig_nodes = await get_node_list(logger, sys_prompt, hum_prompt_nodes, model)
    print(orig_nodes)

    for i, node in enumerate(orig_nodes):
        # breakpoint()
        hum_prompt_node_refine = human_instruction_node_refine.format(node=node)
        orig_nodes[i] = await refine_node(logger, sys_prompt, hum_prompt_node_refine, model)

    hum_prompt_edges = human_instruction_edges.format(graph_nodes=orig_nodes)
    nodes, edges = await get_edge_list(logger, sys_prompt, hum_prompt_edges, model)

    valid = False
    while not valid:
        missed_nodes = check_node_edge_valid(nodes, edges)
        if len(missed_nodes) == 0:
            valid = True
        else:
            print("going into correction mode")
            hum_prompt_corr_edges = human_instruction_corr_edges.format(
                graph_nodes=nodes, edges=edges, missing_nodes=missed_nodes
            )
            nodes, edges = await get_corrected_nodes_edges(
                logger, sys_prompt, hum_prompt_corr_edges, model
            )
    breakpoint()
    # node_list = parse_nodes_and_edges()
    return (nodes, edges)


async def create_graphs_for_graph_store(
    logger: log.InstanceLogger, procs: list[Procedure], model: Model, dataset: str
):
    for i, p in enumerate(procs[32:]):
        graph = await get_graph_from_linear_procedure(logger, p.steps, model, dataset)
