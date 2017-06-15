import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

admission = pd.read_csv("binary.csv")

admissions = admission.loc[np.random.permutation(admission.index)]

num_train = 300
data_train = admission[:num_train]
data_test = admission[num_train:]

model = LogisticRegression()

model.fit(data_train[['gpa','gre','rank']],data_train['admit'])

train = model.predict_proba(data_train[['gpa','gre','rank']])[:,1]

pred_train = model.predict(data_train[['gpa','gre','rank']])

accuracy_train = (pred_train == data_train['admit']).mean()

pred_test = model.predict(data_test[['gpa','gre','rank']])

accuracy_test = (pred_test == data_test['admit']).mean()

from sklearn.metrics import roc_curve, roc_auc_score

train_probs = model.predict_proba(data_train[['gpa', 'gre','rank']])[:,1]
test_probs = model.predict_proba(data_test[['gpa', 'gre','rank']])[:,1]

auc_train = roc_auc_score(data_train["admit"], train_probs)
auc_test = roc_auc_score(data_test["admit"], test_probs)

print('Auc_train: {}'.format(auc_train))
print('Auc_test: {}'.format(auc_test))



