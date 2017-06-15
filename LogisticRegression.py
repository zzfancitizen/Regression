import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

admission = pd.read_csv("binary.csv")

X = admission.iloc[:,1:4]
Y = admission.iloc[:,0]

