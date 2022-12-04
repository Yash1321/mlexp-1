import pandas as pd
import numpy as np

data = np.array(pd.read_csv("candidate-elemination.csv"))
concepts = data[:, :-1]
target = data[:, -1]

len = len(concepts[0])

specific = ["%"]*len
generic = [["?" for i in range(len)] for j in range(len)]

for i, concept in enumerate(concepts):
    # positive output
    if (target[i] == "yes"):
        for j, con in enumerate(concept):
            if (generic[j][j] != con):
                generic[j][j] = con
    # negative output
    else:
        for j, con in enumerate(concept):
            if (specific[j] != con):
                specific[j] = con
