import pandas as pd
import numpy as np

data = np.array(pd.read_csv("find-s.csv", header=0))
concepts = data[:, :-1]
target = data[:, -1]

hypothesis = ["%"] * len(concepts[0])
for i, concept in enumerate(concepts):
    if (target[i] == "yes"):
        hypothesis = concept.copy()
        break

print(concepts, target, hypothesis)
for i, concept in enumerate(concepts):
    if (target[i] == "yes"):
        for j in range(len(concept)):
            if (hypothesis[j] != "?" and hypothesis[j] != concept[j]):
                hypothesis[j] = "?"
    print(hypothesis)
