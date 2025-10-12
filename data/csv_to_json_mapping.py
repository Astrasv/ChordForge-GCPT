import pandas as pd
import json
import ast

df = pd.read_csv("./chords_mapping.csv")

df["Degrees"] = df["Degrees"].apply(ast.literal_eval)

chord_dict = dict(zip(df["Chords"], df["Degrees"]))

with open("./chords_mapping.json", "w") as f:
    json.dump(chord_dict, f, indent=2)
