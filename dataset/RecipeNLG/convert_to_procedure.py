import numpy as np 
import pandas as pd
from dataset.procedure import *
import json
import pickle

csv_path = './recipe_subset.csv'
counter_to_rem = 0

def read_csv(csv_path):
    recipes = pd.read_csv(csv_path)
    # recipes = recipes.head(100)
    # recipes.to_csv('recipe_subset.csv', index=False)
    return recipes

def process_ingredients(detail_ingredients, item_names):
    measuring_keywords=["tbsp", "box", "can", "c.", "cup", "oz", "ml"]
    detail_ing = json.loads(detail_ingredients)

    for ing in detail_ing:
        ing_str_list = ing.split(" ")
        start_idx = 0
        for ing_str in ing_str_list:
            if ing_str.isnumeric() or ing_str[-1] == '.' or 
    # item_list = json.loads(item_names)

    # if len(detail_ing) != len(item_list):
    #     counter_to_rem += 1
    
    # return item_list

def process_steps(directions):
    return json.loads(directions)

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
    
    recipe_procedures = []
    processed = 0

    for index, row in recipes.iterrows():
        _input, _output, _steps = process_recipe(row)
        procedure_obj = create_procedure(_input, _output, _steps)
        recipe_procedures.append(procedure_obj.toJson())
        processed += 1
        if processed%100 == 0:
            print(f"Processed {processed} recipes")

    import pdb; pdb.set_trace()
    #Pickle the list of procedure objects
    with open("recipe_procedures.json", "w") as f:
        json.dump(recipe_procedures, f, indent = 2)


prepare_recipe_memory()



