import pandas as pd
import re
import numpy as np
from Levenshtein import distance as levenshtein
from tqdm import tqdm

                                             
def get_protein_name(text):
    protein_name = text.split("Protein name: ")[1].strip()
    return protein_name
    
ontology = "mf"

testing_file = f"data/{ontology}/valid_data.pkl"
go_terms_file = f"data/{ontology}/terms.pkl"
llama_predictions_file = f"data/{ontology}/predictions.pkl"


output_file = f"data/{ontology}/prediction_matrix.pkl"


print("Loading testing data...")
testing_data = pd.read_pickle(testing_file)
terms_df = pd.read_pickle(go_terms_file)
terms = terms_df['gos'].values.flatten()
proteins_in_testing_data = testing_data["proteins"].values.flatten()
terms_dict = {v: i for i, v in enumerate(terms)}


uniprot_df = pd.read_csv("data/uniprot.tsv", sep="\t")
uniprot_df = uniprot_df[uniprot_df["Entry Name"].isin(proteins_in_testing_data)]
entry_names = uniprot_df["Entry Name"].values.tolist()
protein_names = uniprot_df["Protein names"].values.tolist()
protein_to_entry = {protein_names[i]: entry_names[i] for i in range(len(entry_names))}
entry_to_protein = {entry_names[i]: protein_names[i] for i in range(len(entry_names))}
print(f"Number of proteins in testing data: {len(proteins_in_testing_data)}")
print(f"Number of proteins in uniprot: {len(entry_names)}")
proteins = []
prediction_terms = []
prediction_scores = []


predictions_df = pd.read_pickle(llama_predictions_file)
predictions_df["curated_gene_prot_name"] = predictions_df["gene_prot_name"].apply(get_protein_name)
print(predictions_df.head())
print(entry_to_protein)
not_found = 0
prot_names = set(predictions_df["curated_gene_prot_name"].values.tolist())
for entry_name in proteins_in_testing_data:
    if entry_name in entry_to_protein:
        protein_name = entry_to_protein[entry_name]
        if protein_name in prot_names:
            preds = predictions_df[predictions_df["curated_gene_prot_name"] == protein_name]["prediction"].values[0]
            prediction_scores.append(preds)
        else:
            not_found += 1
            #print(f"Protein {protein_name} not found in predictions")
            prediction_scores.append(np.zeros(len(terms)))
    else:
        not_found += 1
        print(entry_name)
        prediction_scores.append(np.zeros(len(terms)))

print(f"Number of proteins not found: {not_found}")
                        
#print(f"Proteins not found: {not_found}")
testing_data["preds"] = prediction_scores


testing_data.to_pickle(output_file)

