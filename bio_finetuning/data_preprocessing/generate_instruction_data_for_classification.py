import pandas as pd
import json
import os
import random
import re

ontology = "mf"
root = "/data/zhapacfp/llama_adapter/data/"
go_terms_file = f"{root}/{ontology}/terms.pkl"
train_input_file = f"{root}/{ontology}/train_data.pkl"
valid_input_file = f"{root}/{ontology}/valid_data.pkl"
train_output_file = f"{root}/{ontology}/train_with_desc.pkl"
valid_output_file = f"{root}/{ontology}/valid_with_desc.pkl"
terms_df = pd.read_pickle(go_terms_file)
terms = terms_df['gos'].values.flatten()
terms = set(terms)
print(f"Number of terms in terms.pkl: {len(terms)}")

def get_protein_description(gene_name, protein_name, description):
    return f"The gene {gene_name} with protein name {protein_name} is described as {description}."
     

uniprot_file = f"{root}/uniprot.tsv"

train_df = pd.read_pickle(train_input_file)
valid_df = pd.read_pickle(valid_input_file)

train_proteins = train_df["proteins"].tolist()
valid_proteins = valid_df["proteins"].tolist()

all_proteins = train_proteins + valid_proteins

function_columns = ["Gene_Protein", "Gene Ontology (GO)"]

# Load uniprot data
uniprot_df = pd.read_csv(uniprot_file, sep="\t")
uniprot_df = uniprot_df[uniprot_df["Entry Name"].isin(all_proteins)]
print("Loaded {} uniprot entries".format(len(uniprot_df)))
print(f"Columns in uniprot data: {uniprot_df.columns}")

train_data = uniprot_df[uniprot_df["Entry Name"].isin(train_proteins)].fillna("")
train_data = train_data[["Entry Name", "Gene Names", "Protein names", "Function [CC]"]]
valid_data = uniprot_df[uniprot_df["Entry Name"].isin(valid_proteins)].fillna("")
valid_data = valid_data[["Entry Name", "Gene Names", "Protein names", "Function [CC]"]]

train_descriptions = []
print("Generating training descriptions...")
for protein in train_proteins:
    protein_data = train_data[train_data["Entry Name"] == protein]
    try:
        gene_name = protein_data["Gene Names"].values[0]
    except Exception as e:
        gene_name = ""
        print(f"Entry {protein} has no gene name. Assigning empty string.")

    try:
        protein_name = protein_data["Protein names"].values[0]
    except Exception as e:
        protein_name = ""
        print(f"Entry {protein} has no protein name. Assigning empty string.")
        
    try:
        description = protein_data["Function [CC]"].values[0]
    except Exception as e:
        description = ""
        print(f"Entry {protein} has no description. Assigning empty string.")
        
    train_descriptions.append(get_protein_description(gene_name, protein_name, description))
    
train_df["descriptions"] = train_descriptions

valid_descriptions = []
print("Generating validation descriptions...")
for protein in valid_proteins:
    protein_data = valid_data[valid_data["Entry Name"] == protein]

    try:
        gene_name = protein_data["Gene Names"].values[0]
    except Exception as e:
        gene_name = ""
        print(f"Entry {protein} has no gene name. Assigning empty string.")

    try:
        protein_name = protein_data["Protein names"].values[0]
    except Exception as e:
        protein_name = ""
        print(f"Entry {protein} has no protein name. Assigning empty string.")

    try:
        description = protein_data["Function [CC]"].values[0]
    except Exception as e:
        description = ""
        print(f"Entry {protein} has no description. Assigning empty string.")

    valid_descriptions.append(get_protein_description(gene_name, protein_name, description))

valid_df["descriptions"] = valid_descriptions


train_df.to_pickle(train_output_file)
valid_df.to_pickle(valid_output_file)
