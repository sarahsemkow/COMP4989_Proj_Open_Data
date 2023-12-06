from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from functions import *

# file_path = "../data/old/poseangles.csv"
file_path = "../data/200_angles/random_sample_200_all.csv"
# dataframe of all the data
X_train, X_test, y_train, y_test = load_data(file_path)

# Set the range of C values to test
param_grid = {'C': [0.1, 1, 5, 9, 10, 15, 20, 50, 100, 1000]}

# Create an SVM classifier with RBF kernel
svc = SVC(kernel='rbf')

# Perform GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best C value and its corresponding score
best_C = grid_search.best_params_['C']
best_score = grid_search.best_score_

print(f"Best C value: {best_C}")
print(f"Best cross-validation score: {best_score}")

# Get cross-validation scores for different C values
cv_results = grid_search.cv_results_
mean_scores = cv_results['mean_test_score']
std_scores = cv_results['std_test_score']
C_values = param_grid['C']

# Plot cross-validation scores for different C values
plt.errorbar(C_values, mean_scores, yerr=std_scores, fmt='o-', color='b')
plt.title('Cross-validation scores for different C values')
plt.xlabel('C values')
plt.ylabel('Mean cross-validation accuracy')
plt.xscale('log')
plt.grid(True)
plt.show()

# Evaluate the model using the best C value on the test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test set accuracy with best C value: {test_score}")