import json
import random

ontology = "mf"
train_data = f"../data/{ontology}/train_instruction_data.json"
alpaca_data = "../data/alpaca_data.json"

with open(train_data, "r") as f:
    train_data = json.load(f)

with open(alpaca_data, "r") as f:
    alpaca_data = json.load(f)

# merge json files into one

all_data = train_data + alpaca_data
random.shuffle(all_data)

with open(f"../data/{ontology}/all_data.json", "w") as f:
    json.dump(all_data, f, indent=4)
    
