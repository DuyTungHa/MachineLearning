# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# Load the mnist data set. X is the set of explanatory variables and Y is the target variable. 
X, Y = fetch_openml('mnist_784', version = 1, cache = True, return_X_y = True, as_frame = True)

# Explore data shape
print(X.shape)
print(Y.shape)
print(np.unique(Y))

# Discover and visualize the data

# Display one image from each class
classLabels = []
f, axarr = plt.subplots(2,5)
for i in range(2):
  for j in range(5):
    digit_index = np.random.randint(X.shape[0])
    while not classLabels.count(Y.iloc[digit_index]) == 0:
      digit_index = np.random.randint(X.shape[0])

    classLabels.append(Y.iloc[digit_index])
    some_digit = X.iloc[digit_index].to_numpy()
    some_digit_image = some_digit.reshape(28, 28)
    axarr[i,j].imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
    axarr[i,j].axis("off")
plt.show()

# Display the size of each class
print(Y.value_counts())

# Check if there is any outlier (count the number of outliers)
z_scores = np.abs(stats.zscore(X))
print(len(np.where(z_scores > 3)[0]))

# Obtain min, max and mean for each pixel (i.e., each attribute) 
# and display them as an image (say, a mean image)
f, axarr = plt.subplots(1,3)
min_image = X.min().to_numpy().reshape(28, 28)
max_image = X.max().to_numpy().reshape(28, 28)
mean_image = X.mean().to_numpy().reshape(28, 28)
axarr[0].imshow(min_image, cmap = mpl.cm.binary, interpolation="nearest")
axarr[0].axis("off")
axarr[1].imshow(max_image, cmap = mpl.cm.binary, interpolation="nearest")
axarr[1].axis("off")
axarr[2].imshow(mean_image, cmap = mpl.cm.binary, interpolation="nearest")
axarr[2].axis("off")
plt.show()

# Plot a histogram of the intensity of some pixels for different images
X.iloc[:, 0:9].hist()
X.iloc[:, 386:395].hist()
X.iloc[:, 775:784].hist()

# Compute pairwise correlation of columns and print out the top 10 correlated pair of attributes
# https://intellipaat.com/community/20448/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
corr_matrix = X.corr().abs()
sorted_corr = corr_matrix.unstack().sort_values(kind="quicksort")
print(sorted_corr[0:20:2])

# Prepare the data for machine learning algorithms

# Check for missing values
print('Number of missing values: ', X.isnull().sum())

# Normalize the data
scaler = StandardScaler()
X_tr = scaler.fit_transform(X)
random_idx = np.random.randint(X_tr.shape[1])
print('Mean and standard deviation of a random pixel after normalization: ', np.mean(X_tr[:, random_idx]), np.std(X_tr[:, random_idx]))

# Split the data into training set and test set
X_train = X_tr[0:60000]
Y_train = Y.iloc[0:60000].to_numpy()
X_test = X_tr[60000:]
Y_test = Y.iloc[60000:].to_numpy()

# Look at the size of each class in the training and test sets
print(pd.value_counts(Y_train))
print(pd.value_counts(Y_test))

# Select and train models

# Train a Logistic Regression
log_reg = LogisticRegression(max_iter=1500, n_jobs=-1)
log_reg.fit(X_train, Y_train)

# Take the first 5 samples and test them
some_data = X_train[:5]
some_labels = Y_train[:5]
# Print the predicted digit label
print("Predictions:", log_reg.predict(some_data))
# Print the true digit label
print("Labels: ", some_labels)

# Predict with the train model
logreg_predictions = log_reg.predict(X_train)
# Compute the RMSE to evaluate its regression performance
logreg_mse = mean_squared_error(Y_train, logreg_predictions)
logreg_rmse = np.sqrt(logreg_mse)
print('Train Error: ', logreg_rmse)

# Train a decision tree model
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, Y_train)

# Predict with the train model
treereg_predictions = tree_reg.predict(X_train)
# Compute the RMSE to evaluate its regression performance
tree_mse = mean_squared_error(Y_train, treereg_predictions)
tree_rmse = np.sqrt(tree_mse)
print('Train Error: ', tree_rmse)

# Train a Support Vector Machine classifier
svc = SVC()
svc.fit(X_train, Y_train)

# Predict with the train model
svc_predictions = svc.predict(X_train)
# Compute the RMSE to evaluate its regression performance
svc_mse = mean_squared_error(Y_train, svc_predictions)
svc_rmse = np.sqrt(svc_mse)
print('Train Error: ', svc_rmse)

# Fine-tune the model

# Create param grid for models
# https://stackoverflow.com/questions/50265993/alternate-different-models-in-pipeline-for-gridsearchcv
models = {
    'DecisionTreeRegressor': tree_reg,
    'SVC': svc,
    'LogisticRegression': log_reg
}

params = {
    'DecisionTreeRegressor':{ 
        'max_leaf_nodes': list(range(2, 100)),
        'min_samples_split': [2, 3, 4]
    },
    'SVC': {
        'C': [0.1, 1, 10, 100, 1000],  
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
        'kernel': ['poly', 'rbf', 'sigmoid']
    },
    'LogisticRegression': {
        'solver': ['newton-cg', 'sag', 'lbfgs'],
        'multi_class': ['ovr', 'multinomial']
    }  
}

final_models = []
# Fine-tune different models based on params and select the best hyperparameters
for name in models.keys():
    est = models[name]
    est_params = params[name]
    gscv = GridSearchCV(estimator = est, param_grid=est_params, cv=5)
    gscv.fit(X_train[0:1000], Y_train[0:1000])
    print('Best parameters are: {}'.format(gscv.best_estimator_))
    final_models.append(gscv.best_estimator_)

# Evaluate the outcomes

# Predict the target label with the trained model
for final_model in final_models:
  final_predictions = final_model.predict(X_test)
  final_mse = mean_squared_error(Y_test, final_predictions)
  final_rmse = np.sqrt(final_mse)
  print('Final Model: ', final_model)
  print('Report Final Model Error: ', final_rmse)

print('All done!')

# Use the test of statistical significance to evaluate which model is better

# Kfold cross validation
skf = StratifiedKFold(n_splits=10)
# use the fine-tuned version
svc_mdel = final_models[1] 
log_mdel = final_models[2]  
# list of errors for each iteration of k-fold
svc_errs = [] 
log_errs = []

for train_index, val_index in skf.split(X_train, Y_train):
  # train models
  svc_mdel.fit(X_train[train_index], Y_train[train_index])
  log_mdel.fit(X_train[train_index], Y_train[train_index])
  # make predictions on validation set
  svc_pred = svc_mdel.predict(X_train[val_index])
  log_pred = log_mdel.predict(X_train[val_index])
  # compute and save validation errors
  svc_errs.append(np.sqrt(mean_squared_error(Y_train[val_index], svc_pred)))
  log_errs.append(np.sqrt(mean_squared_error(Y_train[val_index], log_pred)))

# Student's t test
meanErrSVC = np.mean(np.array(svc_errs))
meanErrLOG = np.mean(np.array(log_errs))
variance = sum([(log_errs[i] - svc_errs[i] - (meanErrLOG - meanErrSVC))**2 for i in range(10)]) * 0.1
tScore = (meanErrLOG - meanErrSVC) / ((variance / 10)**0.5)
pval = stats.t.sf(np.abs(tScore), 9) * 2
print('p-value: ', pval)

# Whether to reject the null hypothesis based on significance level (0.05)
if pval > 0.05:
    print('Any difference between Support Vector Machine and Logistic Regression is by chance')
else:
    print('Statistically significant difference between Support Vector Machine and Logistic Regression')
