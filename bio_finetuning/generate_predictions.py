import pandas as pd
import re
import numpy as np
from Levenshtein import distance as levenshtein
from tqdm import tqdm
#def process_predictions(text):
#    pattern = r'\[([^\]]+)\]'
#    return re.findall(pattern, text)


go_synonyms = pd.read_csv("data/go_synonyms.tsv", sep="\t")

def find_closest_go_term(go_term):
    go_synonyms["distance"] = go_synonyms["synonym"].apply(lambda x: levenshtein(x, go_term))
    syn = go_synonyms["go"][go_synonyms["distance"].argmin()]
    dist = go_synonyms["distance"].min()
    inverse = 1 / (dist + 1)
    return syn, inverse


def process_predictions(text):
    text = text.replace("### Response:", "")
    functions = text.split(", ")
    gos = []
    dists = []
    for func in functions:
        go_term, dist = find_closest_go_term(func)
        gos.append(go_term)
        dists.append(dist)
    return gos, dists

def get_protein_name(text):
    protein_name = text.split("Protein name: ")[1].strip()
    return protein_name
    
ontology = "mf"

testing_file = f"data/{ontology}/valid_data.pkl"
go_terms_file = f"data/{ontology}/terms.pkl"
llama_predictions_file = f"data/{ontology}/validation_results.txt"
gene_prot_names_file = f"data/{ontology}/validation_gene_prot_names.tsv"
gene_prot_names = pd.read_csv(gene_prot_names_file, sep="\t", header=None)
gene_prot_names.columns = ["name"]
names = gene_prot_names["name"].values.tolist()



output_file = f"data/{ontology}/predictions.pkl"


print("Loading testing data...")
testing_data = pd.read_pickle(testing_file)
terms_df = pd.read_pickle(go_terms_file)
terms = terms_df['gos'].values.flatten()
proteins_in_testing_data = testing_data["proteins"].unique().tolist()
terms_dict = {v: i for i, v in enumerate(terms)}





uniprot_df = pd.read_csv("data/uniprot.tsv", sep="\t")
uniprot_df = uniprot_df[uniprot_df["Entry Name"].isin(proteins_in_testing_data)]
entry_names = uniprot_df["Entry Name"].values.tolist()
protein_names = uniprot_df["Protein names"].values.tolist()
protein_to_entry = {protein_names[i]: entry_names[i] for i in range(len(entry_names))}

proteins = []
prediction_terms = []
prediction_scores = []

with open(llama_predictions_file, "r") as f:
    next_is_protein = False
    lines = f.readlines()
    prot_count = 0
    for line in tqdm(lines, desc="Processing predictions", total=len(lines)):
        if next_is_protein:
            protein_name = get_protein_name(gene_prot_names["name"][prot_count])
            protein_name_from_line = get_protein_name(line)
            assert protein_name.startswith(protein_name_from_line), f"\n{protein_name} != \n {protein_name_from_line}"
            protein = protein_to_entry[protein_name]
            proteins.append(protein)
            next_is_protein = False
            prot_count += 1
            continue
        
        if line.startswith("### Input"):
            next_is_protein = True
            continue
        elif line.startswith("### Response"):
            gos, dists = process_predictions(line)
            prediction_terms.append(gos)
            prediction_scores.append(dists)
                        

pred_terms_dict = dict(zip(proteins, prediction_terms))
pred_scores_dict = dict(zip(proteins, prediction_scores))
preds_term_idx_dict = {k: [terms_dict[v] for v in pred_terms_dict[k] if v in terms_dict] for k in pred_terms_dict }
pred_scores_dict = {k: [v for i, v in enumerate(pred_scores_dict[k]) if pred_terms_dict[k][i] in terms_dict] for k in pred_scores_dict}

pred_matrix = np.zeros((len(testing_data), len(terms)))

not_found = 0
for i, p in enumerate(testing_data['proteins'].values):
    if p in pred_terms_dict:
        pred_matrix[i, preds_term_idx_dict[p]] = pred_scores_dict[p]
 
    else:
        not_found += 1

print(f"Proteins not found: {not_found}")
testing_data["preds"] = pred_matrix.tolist()


testing_data.to_pickle(output_file)

