import mowl
mowl.init_jvm("10g")

from mowl.datasets import PathDataset
from mowl.corpus import extract_annotation_corpus
from tqdm import tqdm
import json

ontology_file = "../data/go-plus.owl"
output_file = "../data/go_descriptions.json"
output_syn_file = "../data/go_synonyms.json"

dataset = PathDataset(ontology_file)
ontology = dataset.ontology

annots = extract_annotation_corpus(ontology)

desc_pattern = " <http://purl.obolibrary.org/obo/IAO_0000115> "
syn_pattern = " <http://www.geneontology.org/formats/oboInOwl#hasExactSynonym> "

descriptions = [x for x in annots if desc_pattern in x and "GO_" in x]
descriptions = [x.split(desc_pattern) for x in descriptions]
descriptions = [ (x[0].split("/")[-1].replace("_",":"), x[1].strip()) for x in descriptions]

synonyms = [x for x in annots if syn_pattern in x and "GO_" in x]
synonyms = [x.split(syn_pattern) for x in synonyms]
synonyms = [ (x[0].split("/")[-1].replace("_",":"), x[1].strip()) for x in synonyms]




instructions = []
instruct_desc = "What is the description of the following Gene Ontology term?"
for x in tqdm(descriptions, desc="Generating descriptions", total=len(descriptions)):
    go, desc = tuple(x)
    go = go[1:] if go.startswith("<") else go
    go = go[:-1] if go.endswith(">") else go
    instructions.append((instruct_desc, go, desc))

instruct_syn = "What is a synonym of the following Gene Ontology term?"
for x in tqdm(synonyms, desc="Generating synonyms", total=len(synonyms)):
    go, syn = tuple(x)
    go = go[1:] if go.startswith("<") else go
    go = go[:-1] if go.endswith(">") else go
    instructions.append((instruct_syn, go, syn))


    
keys = ["instruction", 'input', 'output']
train_go_descriptions = [dict(zip(keys, x)) for x in instructions]

with open(output_file, "w") as f:
    json.dump(train_go_descriptions, f, indent=4)
