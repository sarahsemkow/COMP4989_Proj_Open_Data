import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV

from functions import *

file_path = "../data/200_angles/random_sample_200_all.csv"
# dataframe of all the data
X_train, X_test, y_train, y_test = load_data(file_path)

# Set the range of n_neighbors values to test
param_grid = {'n_neighbors': np.arange(1, 21)}  # Testing neighbors from 1 to 20

# Create a KNN classifier
knn = KNeighborsClassifier()

# Perform GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best n_neighbors value and its corresponding score
best_n_neighbors = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_

print(f"Best n_neighbors value: {best_n_neighbors}")
print(f"Best cross-validation score: {best_score}")

# Get cross-validation scores for different n_neighbors values
cv_results = grid_search.cv_results_
mean_scores = cv_results['mean_test_score']
std_scores = cv_results['std_test_score']
n_neighbors_values = param_grid['n_neighbors']

# Plot cross-validation scores for different n_neighbors values
plt.errorbar(n_neighbors_values, mean_scores, yerr=std_scores, fmt='o-', color='b')
plt.title('Cross-validation scores for different n_neighbors values')
plt.xlabel('n_neighbors')
plt.ylabel('Mean cross-validation accuracy')
plt.xticks(np.arange(1, 21, 2))  # Show ticks for every 2 neighbors
plt.grid(True)
plt.show()

# Evaluate the model using the best n_neighbors value on the test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test set accuracy with best n_neighbors value: {test_score}")