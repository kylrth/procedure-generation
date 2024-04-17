import champ_dataset
from dataset.base import Procedure

def load_dataset():
    dataset = champ_dataset.load('v0')
    return dataset.problems, dataset.hcd ints, dataset.concepts

def get_input(content, hints, concepts):
    input_str = ''
    
    #Adding category now
    input_str += f'Category: {content.category}\n'

    concept_list = []
    hints_list = []

    for elem in content.ch_list:
        if elem[0] == 'C':
            concept_list.append(concepts[elem]._text)
        elif elem[0] == 'H':
            hints_list.append(hints[elem]._text)
    
    #Adding concepts now
    input_str += f'Concepts: {str(concept_list)}\n'

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


if __name__=="__main__":
    probs, hints, concepts = load_dataset()
    procedure_list = parse_problems(probs, hints, concepts)


    