import numpy as np
import pandas as pd
from dataset.base import Procedure
import json
import pickle

csv_path = './recipe_nlg.csv'
counter_to_rem = 0

def read_csv(csv_path):
    recipes = pd.read_csv(csv_path)
    return recipes

def process_ingredients(detail_ingredients, item_names):
    detail_ing = json.loads(detail_ingredients)  
    return detail_ing

def process_steps(directions):
    steps =  json.loads(directions)
    return steps

def create_procedure(_input, _output, _steps):
    procedure_obj = Procedure()
    procedure_obj._set_input(_input)
    procedure_obj._set_output(_output)
    procedure_obj._set_steps(_steps)
    return procedure_obj

def process_recipe(df_row):
    _input = process_ingredients(df_row['ingredients'], df_row['NER'])
    _output = df_row['title']
    _steps = process_steps(df_row['directions'])
    return _input, _output, _steps

def prepare_recipe_memory():
    global csv_path, counter_to_rem
    #Read Recipes CSV
    recipes = read_csv(csv_path) #Returns a pandas dataframe
    print("Dataset read now!")
    recipe_procedures = []
    processed = 0

    for index, row in recipes.iterrows():
        _input, _output, _steps = process_recipe(row)
        procedure_obj = create_procedure(_input, _output, _steps)
        recipe_procedures.append(procedure_obj.to_json())
        processed += 1
        if processed%1000 == 0:
            print(f"Processed {processed} recipes")

    #Pickle the list of procedure objects
    with open("recipe_procedures.json", "w") as f:
        json.dump(recipe_procedures, f, indent = 2)


prepare_recipe_memory()
