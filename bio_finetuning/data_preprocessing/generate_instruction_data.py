import pandas as pd
import json
import os
import random

def get_question(protein_name):
    question = random.choice(questions_df[0])
    return question.replace("<PROTEIN NAME>", protein_name)


questions_file = "data/questions.tsv"
uniprot_file = "data/uniprot.tsv"
train_output_file = "data/train_instruction_data.json"
valid_output_file = "data/valid_instruction_data.json"

train_proteins_file = "data/mf/train_proteins.csv"
valid_proteins_file = "data/mf/valid_proteins.csv"

train_proteins_df = pd.read_csv(train_proteins_file, header=None)
valid_proteins_df = pd.read_csv(valid_proteins_file, header=None)

all_proteins = set(train_proteins_df[0]).union(set(valid_proteins_df[0]))

# Load questions
questions_df = pd.read_csv(questions_file, sep="\t", header=None)
print("Loaded {} questions".format(len(questions_df)))


# Load uniprot data
uniprot_df = pd.read_csv(uniprot_file, sep="\t")
print("Loaded {} uniprot entries".format(len(uniprot_df)))
print(f"Columns in uniprot data: {uniprot_df.columns}")

uniprot_df_with_functions = uniprot_df[uniprot_df["Gene Ontology (GO)"].notna()]
print(f"Found {len(uniprot_df_with_functions)} entries with GO functions")

# Filter out proteins that are not in the training or validation set
uniprot_experimental = uniprot_df_with_functions[uniprot_df_with_functions["Entry Name"].isin(all_proteins)]
print(f"Found {len(uniprot_experimental)} entries with experimental annotations")


#Training set

# Get Function [CC] column where Entry Name is in the training set
train_contexts = uniprot_experimental[uniprot_experimental["Entry Name"].isin(train_proteins_df[0])][["Function [CC]"]]
train_proteins = uniprot_experimental[uniprot_experimental["Entry Name"].isin(train_proteins_df[0])]["Entry Name"]
train_queries = [get_question(protein_name) for protein_name in train_proteins]

train_outputs = uniprot_experimental[uniprot_experimental["Entry Name"].isin(train_proteins_df[0])][["Gene Ontology (GO)"]]
     
train_keys = ["Context", "Query", "Outputs"]

train_data = zip(train_contexts["Function [CC]"], train_queries, train_outputs["Gene Ontology (GO)"])
train_data = [dict(zip(train_keys, x)) for x in train_data]
print(f"Generated {len(train_data)} training examples")
train_json_data = json.dumps(train_data, indent=4)

with open(train_output_file, "w") as f:
    f.write(train_json_data)


#Validation set

valid_contexts = uniprot_experimental[uniprot_experimental["Entry Name"].isin(valid_proteins_df[0])][["Function [CC]"]]
valid_proteins = uniprot_experimental[uniprot_experimental["Entry Name"].isin(valid_proteins_df[0])]["Entry Name"].tolist()
valid_queries = [get_question(protein_name) for protein_name in valid_proteins]

valid_outputs = uniprot_experimental[uniprot_experimental["Entry Name"].isin(valid_proteins_df[0])][["Gene Ontology (GO)"]]

valid_keys = ["Context", "Query", "Outputs"]

valid_data = zip(valid_contexts["Function [CC]"], valid_queries, valid_outputs["Gene Ontology (GO)"])
valid_data = [dict(zip(valid_keys, x)) for x in valid_data]
print(f"Generated {len(valid_data)} validation examples")
valid_json_data = json.dumps(valid_data, indent=4)

with open(valid_output_file, "w") as f:
    f.write(valid_json_data)
