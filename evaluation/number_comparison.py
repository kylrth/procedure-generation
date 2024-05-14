from dataset import Procedure
from systems import Model
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
import re

def get_numbers_list(model: Model, proc_steps: str):
    prompt = [
            SystemMessage(content="Please extract the numbers along with their units only from the procedure given below. For example, 'preheat the oven to 350°' gives '350° C'. However, do not extract numbers which are description of the crockery and also avoid repetition of numbers in your response."),
            HumanMessage(content=f"{proc_steps}\n Your response should be a python list starting with [ and end with ]."),
            ]
    
    number_list = model.generate(prompt)
    list_items = re.sub(r'[\s]', '', number_list)[1:-1].split(',')
    return list_items

def edit_distance(model: Model, gold: Procedure, generated: str):
    gold_proc_steps = str(gold.steps)[1:-1] #Removing the brackets    
    gold_numbers_list = get_numbers_list(model, gold_proc_steps)
    
    generated_numbers_list = get_numbers_list(model, generated)
    
    num_present = 0
    for number in gold_numbers_list:
        if number in generated_numbers_list:
            num_present += 1
    
    return (num_present/len(gold_numbers_list))