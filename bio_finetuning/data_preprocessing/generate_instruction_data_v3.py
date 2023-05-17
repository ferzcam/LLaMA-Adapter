import pandas as pd
import json
import os
import random
import re

ontology = "mf"

def get_pf_query():
    return "What are the Gene Ontology terms of the functions of the following gene/protein?"

def get_protein_description_query():
    return "What is the description of the following gene/protein?"

def remove_go_ids(text):
    new_text = re.sub(r'\[.*?\]', '', text)
    new_text = new_text.replace(" ;", ",")
    return new_text


#questions_file = "../data/questions.tsv"

uniprot_file = "../data/uniprot.tsv"
go_descriptions = "../data/go_descriptions.json"

train_output_file = f"../data/{ontology}/train_instruction_data.json"
valid_output_file = f"../data/{ontology}/valid_instruction_data.json"



train_data_file = f"../data/{ontology}/train_data.pkl"
valid_data_file = f"../data/{ontology}/valid_data.pkl"
train_df = pd.read_pickle(train_data_file)
valid_df = pd.read_pickle(valid_data_file)

train_proteins = train_df["proteins"].unique().tolist()
valid_proteins = valid_df["proteins"].unique().tolist()

all_proteins = train_proteins + valid_proteins

# Load questions
#questions_df = pd.read_csv(questions_file, sep="\t", header=None)
#print("Loaded {} questions".format(len(questions_df)))

description_columns = ["Gene_Protein", "Function [CC]"]
function_columns = ["Gene_Protein", "Gene Ontology (GO)"]

# Load uniprot data
uniprot_df = pd.read_csv(uniprot_file, sep="\t")
uniprot_df = uniprot_df[uniprot_df["Entry Name"].isin(all_proteins)]
print("Loaded {} uniprot entries".format(len(uniprot_df)))
print(f"Columns in uniprot data: {uniprot_df.columns}")

train_data = uniprot_df[uniprot_df["Entry Name"].isin(train_proteins)].fillna("")
train_data["Gene_Protein"] = "Gene name: " + train_data["Gene Names"] + " Protein name: " + train_data["Protein names"]
valid_data = uniprot_df[uniprot_df["Entry Name"].isin(valid_proteins)].fillna("")
valid_data["Gene_Protein"] = "Gene name: " + valid_data["Gene Names"] + " Protein name: " + valid_data["Protein names"]

train_prot_descriptions = train_data[description_columns]
train_prot_descriptions["Function [CC]"] = train_prot_descriptions["Function [CC]"].apply(lambda x: "The description is: " + x.replace("FUNCTION: ", ""))
train_prot_functions = train_data[function_columns]
train_prot_functions["Gene Ontology (GO)"] = train_prot_functions["Gene Ontology (GO)"].apply(lambda x: remove_go_ids(x))


valid_prot_descriptions = valid_data[description_columns]
valid_prot_descriptions["Function [CC]"] = valid_prot_descriptions["Function [CC]"].apply(lambda x: "The description is: " + x.replace("FUNCTION: ", ""))
valid_prot_functions = valid_data[function_columns]
valid_prot_functions["Gene Ontology (GO)"] = valid_prot_functions["Gene Ontology (GO)"].apply(lambda x: remove_go_ids(x))

keys = ["instruction","input", "output"]

prot_description_queries = [get_protein_description_query() for _ in range(len(all_proteins))]
function_prediction_queries = [get_pf_query() for _ in range(len(all_proteins))]

train_description_data = zip(prot_description_queries, train_prot_descriptions["Gene_Protein"], train_prot_descriptions["Function [CC]"])
valid_description_data = zip(prot_description_queries, valid_prot_descriptions["Gene_Protein"], valid_prot_descriptions["Function [CC]"])
description_data = list(train_description_data) + list(valid_description_data)

train_function_prediction_data = zip(function_prediction_queries, train_prot_functions["Gene_Protein"], train_prot_functions["Gene Ontology (GO)"])
train_data = list(train_function_prediction_data) + description_data
train_data = [dict(zip(keys, x)) for x in train_data]

go_description_data = json.load(open(go_descriptions, "r"))
#go_description_data = []
train_data = train_data + go_description_data
random.shuffle(train_data)

print(f"Generated {len(train_data)} training examples")
train_json_data = json.dumps(train_data, indent=4)

with open(train_output_file, "w") as f:
    f.write(train_json_data)


valid_function_prediction_data = zip(function_prediction_queries, valid_prot_functions["Gene_Protein"], valid_prot_functions["Gene Ontology (GO)"])
valid_data = valid_function_prediction_data
valid_data = [dict(zip(keys, x)) for x in valid_data]
print(f"Generated {len(valid_data)} validation examples")
valid_json_data = json.dumps(valid_data, indent=4)

with open(valid_output_file, "w") as f:
    f.write(valid_json_data)
