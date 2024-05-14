from dataset import Procedure
from systems import Model
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
import re
import editdistance as edt

def get_one_word_list(model: Model, proc_steps: str):
    prompt = [
            SystemMessage(content="Please replace each step in the given procedure with a one-word description of the action performed in that step. For example, 'mix flour and eggs in a bowl' will be replaced with 'mixing'."),
            HumanMessage(content=f"{proc_steps}\n Your response should be a python list starting with [ and end with ]."),
            ]
    
    one_word_list = model.generate(prompt)
    list_items = re.sub(r'[\s]', '', one_word_list)[1:-1].split(',')
    return list_items

def edit_distance(model: Model, gold: Procedure, generated: str):
    gold_proc_steps = str(gold.steps)[1:-1] #Removing the brackets    
    gold_one_word_list = get_one_word_list(model, gold_proc_steps)
    
    generated_one_word_list = get_one_word_list(model, generated)
    
    edit_distance = edt.eval(generated_one_word_list, gold_one_word_list)
    
    return edit_distance