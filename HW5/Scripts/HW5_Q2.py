import math
import numpy as np
import collections
import pandas as pd
import statsmodels.api as sm
from sklearn.cross_validation import KFold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

train_set = pd.read_csv("../HW5_jobs_data/train_data.csv")
print "Dimensions \n" + "Training :" + str(train_set.shape)

train_set['Employer'] = train_set['Employer'].astype('category')
categorical_cols = train_set.select_dtypes(['category']).columns

train_set[categorical_cols] = train_set[categorical_cols].apply(lambda x: x.cat.codes)

train_data = train_set[list(train_set.columns[2:270])]
test_data = train_set[list(train_set.columns[270:271])]
test_data = test_data.replace(['nonDS', 'DS'], [0, 1])

selector = SelectKBest(chi2, k=10)
selector.fit(train_data, np.ravel(np.array(test_data)))
mask = selector.get_support(True)
mask_columns = list(train_data.iloc[:, mask].columns)
print mask_columns

file = open("top_10_features.csv", "wb")
for val in mask_columns:
    file.write(val + "\n")
file.close()