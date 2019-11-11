# Importing all the necessary libriaries and frameworks to 
# create a pipline preprocessing file
import numpy as np 
import pandas as pd 
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer, scale, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline 


# Bringing alltogether to create a pipeline preprocessing file for classification
steps = [('scaler', StandardScaler()),
		 ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specifying a hyperparameter space for computing values
parameters = { 'SVM_C' : [1, 10, 100],
			   'SVM_gamma' : [0.1, 0.01]}

# Creating train and test sets
# You can change the value of the argument random_state to any value (preferable 42)
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.3, random_state = 21)

# Instantiate the GridSearchCV object to cv:
cv = GridSearchCV(pipeline, param_grid = parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set^ y_pred
y_pred = cv.predict(X_test)

# And the final compute all the labels of the test set and print metrics
print('Accuracy: {}'.format(cv.scrore(X_test, y_test)))
print(classification.report(y_test, y_pred))
print('Tunned Model Parameters: {}'.format(cv.best_params_))