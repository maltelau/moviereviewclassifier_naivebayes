#!/usr/bin/env python3

from naivebayes import *
import pandas as pd
import sys
import numpy as np


if len(sys.argv) == 1:
    X = pd.Series(input("Write your review here and press [ENTER] to predict the label:\n"))
    
else:
    X = pd.Series(' '.join(sys.argv[1:]))

    
# print(f"Review:\n{X[0]}\n")

model, pipeline = load_model()
for step in pipeline:
    X = step.transform(X)

Y = model.predict(X)
posterior = np.exp(model.posterior)
posterior = (posterior / posterior.sum(axis = 1)).max() * 100
print(f'Predicted class: {Y[0]}\nPosterior probability: {posterior:.1f}%')
