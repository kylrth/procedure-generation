import csv
import json
import champ_dataset
from os import PathLike
from typing import Any
from .base import Dataset, Doc, Procedure

def load_dataset():
    dataset = champ_dataset.load('v0')
    return dataset.problems, dataset.hints, dataset.concepts

def get_input(content, hints, concepts):
    input_str = ''
    
    #Adding category now
    input_str += f'Category: {content.category}\n'

    # concept_list = []
    hints_list = []

    for elem in content.ch_list:
        # if elem[0] == 'C':
        #     concept_list.append(concepts[elem]._text)
        if elem[0] == 'H':
            hints_list.append(hints[elem]._text)
    
    # # Adding concepts now
    # input_str += f'Concepts: {str(concept_list)}\n'

    #Adding hints now
    input_str += f'Hints: {str(hints_list)}'

    return input_str

def get_output(content):
    return content.text

def get_solution_steps(content):
    steps_list = []
    for step in content.solution.steps:
        steps_list.append(step.text)
    
    steps_list.append(f'The answer is {content.answer}')
    return steps_list

def make_procedure_object(content, hints, concepts):
    input_str = get_input(content, hints, concepts)
    output_str = get_output(content)
    step_list = get_solution_steps(content)
    procedure_obj = Procedure(input_str, output_str, step_list)
    return procedure_obj

def parse_problems(probs, hints, concepts):
    procedure_list = []
    for _, content in probs.items():
        proc_obj = make_procedure_object(content, hints, concepts)
        procedure_list.append(proc_obj)
    
    return procedure_list

#Pass data dir as None for CHAMP dataset
class CHAMP(Dataset):
    def __init__(self, data_dir: str | PathLike, n: int | None = None):
        super().__init__(data_dir)
        self.n = n

    def _init_procedures(self) -> list[Procedure]:
        out = []
        probs,hints,concepts = load_dataset()        
        out = parse_problems(probs, hints, concepts)
        return out

    def _get_docs(self) -> list[Doc]:
        doc_list = []
        _,_,concepts = load_dataset()
        for key, val in concepts.items():
            doc_list.append(Doc(title=key, contents=val._text))
        
        return doc_list
