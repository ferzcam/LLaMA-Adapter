import pandas as pd
import json
import os
import random

ontology = "mf"

def get_pf_query():
    return "What are the Gene Ontology terms of the functions of the following protein?"

def get_protein_description_query():
    return "What is the description of the following protein?"

questions_file = "../data/questions.tsv"
uniprot_file = "../data/uniprot.tsv"
train_output_file = f"../data/{ontology}/train_instruction_data.json"
valid_output_file = f"../data/{ontology}/valid_instruction_data.json"

train_proteins_file = f"../data/{ontology}/train_proteins.csv"
valid_proteins_file = f"../data/{ontology}/valid_proteins.csv"

train_proteins_df = pd.read_csv(train_proteins_file, header=None)
valid_proteins_df = pd.read_csv(valid_proteins_file, header=None)


all_proteins = set(train_proteins_df[0]).union(set(valid_proteins_df[0]))

# Load questions
#questions_df = pd.read_csv(questions_file, sep="\t", header=None)
#print("Loaded {} questions".format(len(questions_df)))


# Load uniprot data
uniprot_df = pd.read_csv(uniprot_file, sep="\t")
print("Loaded {} uniprot entries".format(len(uniprot_df)))
print(f"Columns in uniprot data: {uniprot_df.columns}")

uniprot_df_with_functions = uniprot_df[uniprot_df["Gene Ontology (GO)"].notna()]
print(f"Found {len(uniprot_df_with_functions)} entries with GO functions")

# Filter out proteins that are not in the training or validation set
uniprot_experimental = uniprot_df_with_functions[uniprot_df_with_functions["Entry Name"].isin(all_proteins)]
uniprot_experimental["Function [CC]"] = uniprot_experimental["Function [CC]"].fillna("")
print(f"Found {len(uniprot_experimental)} entries with experimental annotations")


#Training set
# Get Function [CC] column where Entry Name is in the training set
train_contexts = uniprot_experimental[uniprot_experimental["Entry Name"].isin(train_proteins_df[0])][["Function [CC]"]]
train_contexts["Function [CC]"] = train_contexts["Function [CC]"].apply(lambda x: "The description is: " + x.replace("FUNCTION: ", ""))
train_proteins = uniprot_experimental[uniprot_experimental["Entry Name"].isin(train_proteins_df[0])]["Entry Name"]
#train_pf_queries = [get_question(protein_name) for protein_name in train_proteins]
train_outputs = uniprot_experimental[uniprot_experimental["Entry Name"].isin(train_proteins_df[0])][["Gene Ontology (GO)"]]
train_outputs["Gene Ontology (GO)"] = train_outputs["Gene Ontology (GO)"].apply(lambda x: "The GO terms associated with that protein are: " + x)

#Validation set
valid_contexts = uniprot_experimental[uniprot_experimental["Entry Name"].isin(valid_proteins_df[0])][["Function [CC]"]]
valid_contexts["Function [CC]"] = valid_contexts["Function [CC]"].apply(lambda x: "The description is: " + x.replace("FUNCTION: ", ""))
valid_proteins = uniprot_experimental[uniprot_experimental["Entry Name"].isin(valid_proteins_df[0])]["Entry Name"].tolist()
#valid_queries = [get_question(protein_name) for protein_name in valid_proteins]
valid_outputs = uniprot_experimental[uniprot_experimental["Entry Name"].isin(valid_proteins_df[0])][["Gene Ontology (GO)"]]
valid_outputs["Gene Ontology (GO)"] = valid_outputs["Gene Ontology (GO)"].apply(lambda x: "The GO terms associated with that protein are: " + x)
#train_prot_description_queries = [get_protein_description_query(protein_name) for protein_name in all_proteins]

keys = ["instruction","input", "output"]

prot_description_queries = [get_protein_description_query() for _ in range(len(all_proteins))]

train_description_data = zip(prot_description_queries, train_proteins, train_contexts["Function [CC]"])
valid_description_data = zip(prot_description_queries, valid_proteins, valid_contexts["Function [CC]"])
description_data = list(train_description_data) + list(valid_description_data)

function_prediction_queries = [get_pf_query() for _ in range(len(all_proteins))]
train_function_prediction_data = zip(function_prediction_queries, train_proteins, train_outputs["Gene Ontology (GO)"])

train_data = list(train_function_prediction_data) + description_data
random.shuffle(train_data)
train_data = [dict(zip(keys, x)) for x in train_data]
print(f"Generated {len(train_data)} training examples")
train_json_data = json.dumps(train_data, indent=4)

with open(train_output_file, "w") as f:
    f.write(train_json_data)


valid_function_prediction_data = zip(function_prediction_queries, valid_proteins, valid_outputs["Gene Ontology (GO)"])
valid_data = valid_function_prediction_data
valid_data = [dict(zip(keys, x)) for x in valid_data]
print(f"Generated {len(valid_data)} validation examples")
valid_json_data = json.dumps(valid_data, indent=4)

with open(valid_output_file, "w") as f:
    f.write(valid_json_data)
