import pandas as pd
import collections
import statsmodels.api as sm
import math
from sklearn.cross_validation import KFold
import numpy as np

train_set = pd.read_csv("../HW5_jobs_data/train_data.csv")
test_set = pd.read_csv("../HW5_jobs_data/test_data.csv")

print "Dimensions \n" + "Training :" + str(train_set.shape)
print "Testing :" + str(test_set.shape)

train_set['Employer'] = train_set['Employer'].astype('category')
categorical_cols = train_set.select_dtypes(['category']).columns

test_set['Employer'] = test_set['Employer'].astype('category')
categorical_cols = test_set.select_dtypes(['category']).columns

train_set[categorical_cols] = train_set[categorical_cols].apply(lambda x: x.cat.codes)
test_set[categorical_cols] = test_set[categorical_cols].apply(lambda x: x.cat.codes)

train_data = train_set[list(train_set.columns[2:270])]
test_data = train_set[list(train_set.columns[270:271])]
test_data = test_data.replace(['nonDS', 'DS'], [0, 1])
test_set = test_set[list(test_set.columns[2:270])]

def perform_regression(x_train, x_test, y_train, y_test, fold):
    y_train = collections.deque(y_train)
    y_train.rotate(-1)
    y_train = list(y_train)
    x_train = list(x_train)
    result = sm.OLS(y_train, x_train).fit()
    test_prediction = result.predict(x_test)
    for x in range(0, len(y_test)):
        avg_error = math.fabs(y_test[x] - test_prediction[x])
    print "Average Error Fold", fold, avg_error

n_folds = 10
fold = 1

print "Performing 10-Fold Cross Validation"

X = np.array(train_data)
Y = np.array(test_data)
k_folds = KFold(len(X), n_folds=10, shuffle=False, random_state=None)

for train_index, test_index in k_folds:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    # performing linear regression
    perform_regression(X_train, X_test, y_train, y_test, fold)
    fold += 1

count1 = 0
count2 = 0
final_model = sm.OLS(test_data, train_data).fit()
final_test_prediction = final_model.predict(test_set)

result = []
for i in range(0, len(final_test_prediction)):
    if final_test_prediction[i] < 0.5:
        result.append("nonDS")
    else:
        result.append("DS")

file = open("test_predictions.csv", "wb")
for val in result:
    file.write(val + "\n")
file.close()

print result.count("nonDS")
print result.count("DS")