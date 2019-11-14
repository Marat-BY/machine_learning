# NDA project the preprocessing and model 
# building code are only available

# Importing libriaries
import numpy as np 
import pandas as pd 

# Loading DataFrame
df = pd.read_csv('TrainingData.csv')

# Checking the dtypes and data distribution of df
df.info()

""" 
The result:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1560 entries, 0 to 1559
Data columns (total 26 columns)
Index: 			1560 non-null items int64
Function:  		1560 non-null items object
Use:			1560 non-null items object
Sharing:        1560 non-null items object
Reporting:		1560 non-null items object
Student_Type:	1560 non-null items object
..........................................
dtypes: float64(2), int64(1), object(23)
memory usage: 24.3+ Mb
"""

# Summary statistics for numerical data types
df.describe()

# Descriptive visual statistics
import matplotlib.pyplot as plt 
# Creating a histogram
plt.hist(df['FTE'].dropna())

# Title and labels
plt.title('distribution of %full-time employees work')
plt.xlabel('% of full-time')
plt.ylabel('number of employees')
# Display plot
plt.show();

""" 
                  FTE         Total
    count  449.000000  1.542000e+03
    mean     0.493532  1.446867e+04
    std      0.452844  7.916752e+04
    min     -0.002369 -1.044084e+06
    25%           NaN           NaN
    50%           NaN           NaN
    75%           NaN           NaN
    max      1.047222  1.367500e+06

"""
# The high variance in expenditures makes sense
# It looks like the FTE column is bimodal distributed
# There are some part-time and some full time employees.

# As a DataFrame contains a lot of slow object types
# It is necessary to do some type of conversion
LABELS = ['Function', 'Use', 'Sharing', 'Reporting',
		  'Student_Type', 'Position_Type', 'Object_Type',
		  'Pre_K', 'Operating_Status']

categorize_label = lambda x: x.astype('category')
df[LABELS] = df[LABELS].apply(categorize_label, axis = 0)
print(df[LABELS].dtypes)

# Counting unique labels
num_unique_labels = df[LABELS].apply(pd.Series.nunique)
num_unique_labels.plot(kind = 'bar')
plt.xlabel('labels')
plt.ylabel('Number of unique labels')
plt.show();
# http://prntscr.com/pvkzjt

# Calculating LogLoss function over the data
def compute_log_loss(predicted, actual, eps = 1e - 14):
	""" 
	 Calculation of the logarithmic loss between predicted
	 and actual data when these are 1D arrays.

	 :param predicted: The predicted probabilities as floats between 0 - 1
	 :param actual: The actual binary labels. Either 0 or 1
	 :param eps (optional): log(0) is inf so we need to offset preicted values 
	 						slightly by eps from 0 to 1.
	"""
	preicted = np.clip(preicted, eps, 1 - eps)

	loss = -1 * np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

	return loss
# Computing All The Log Loss function to all labeled and predicted data
correct_confident_loss = compute_log_loss(correct_confident, actual_labels)
print('Log loss, correct and confident: {}'.format(correct_confident_loss))

# Log Loss for the 2nd case:
correct_not_confident_loss = compute_log_loss(correct_not_confident, actual_labels)

# Log Loss for 3rd case:
wrong_not_confident_loss = compute_log_loss(wrong_not_confident, actual_labels)

# Log Loss for 4th case:
wrong_confident_loss = compute_log_loss(wrong_confident, actual_labels)

# Log Loss for actual labels
actual_labels_loss = compute_log_loss(actual_labels, actual_labels)
print('Log loss, actual_labels: {}'.format(actual_labels, actual_labels))

""" 
Log loss, correct and confident: 0.05129329438755058
Log loss, actual_labels: 9.99200722162646e-15

Log loss, correct and confident: 0.05129329438755058
Log loss, actual_labels: 9.99200722162646e-15

Log loss, correct and confident: 0.05129329438755058
Log loss, correct and not confident: 0.4307829160924542
Log loss, wrong and not confident: 1.049822124498678
Log loss, wrong and confident: 2.9957322735539904
Log loss, actual labels: 9.99200722162646e-15

"""
# Splitting data into train and test when labels do not have enough sample size in DF
from warnings import warn 

def multilabel_sample(y, size = 1000, min_cout = 5, seed = None):
	"""
		Takes a matrix of binary labels 'y' and returns
		the indices for a sample of size 'size' if 'size' > 1
		or 'size' * len(y) if size -< 1
	"""
	try:
		if (np.unique(y).astype(int) != np.array([0, 1])).any():
			raise ValueError()
	except (TypeError, ValueError):
		raise ValueError('multilabel_sample only works with binary indicator matrices')

	if (y.sum(axis = 0) < min_count).any():
		raise ValueError('Some classes do not have enough examples. Change min_count if necessary.')

	if size <= 1:
		size = np.floor(y.shape[0] * size)

	if y.shape[1] * min_count > size:
		msg = 'Size less than number of columns * min_count, returning {} items instead of {}.'
		warn(msg.format(y.shape[1] * min_count, size))
		size = y.shape[1] * min_count

	rng = np.random.RandomState(seed if seed is not None else np.random.randint(1))

	if isinstance(y, pd.DataFrame):
		choices = y.index 
		y = y.values
	else:
		choices = np.arange(y.shape[0])

	sample_idxs = np.array([], dtype = choices.dtype)

	# first, guarantee > mint_count of each label
	for j in range(y.shape[1]):
		label_choices = choices[y[:, j] == 1]
		label_idxs_sampled = rng.choice(label_choices, size = min_count, replace = False)
		sample_idxs = np.concatenate([label_idxs_sampled, sample_idxs])

	sample_idxs = np.unique(sample_idxs)

	# we have at least min_count of each we can just random sample
	sample_count = int(size - sample_idxs.shape[0])

	# get sample_count indices from remaining choices
	remaining_choices = np.setdiff1d(choices, sample_idxs)
	remaining_sampled = rng.choice(remaining_choices,
								   size = sample_count,
								   replace = False)

	return np.concatenate([sample_idxs, remaining_sampled])


def multilabel_sample_dataframe(df, labels, size, min_count = 5, seed = None):
	"""
		Takes a DataFrame 'df' and returns a sample size of 'size' where all
		classes in the binary matrix 'labels' are represented at least 'min_count' times
	"""
	idxs = multilabel_sample(labels, size = size, min_count = min_count, seed = seed)
	return df.loc[idxs]


def multilabel_train_test_split(X, Y, size, min_count = 5, seed = seed):
	"""
		Takes a features matrix 'X' and label matrix 'Y' and returns 
		(X_train, X_test, Y_train, Y_test) where all classes in Y are
		represented at least 'min_count' times
	"""
	index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])

	test_set_idxs = multilabel_sample(Y, size = size, min_count = min_count, seed = seed)
	train_set_idxs = np.setdiff1d(index, test_set_idxs)

	test_set_mask = index.isin(test_set_idxs)
	train_set_mask = ~test_set_mask

	return(X[train_set_mask], X[test_set_mask], Y[train_set_mask], Y[test_set_mask])


NUMERIC_COLUMNS = ['FTE', 'Total']

# Numeric data only dataframe:
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

# Retrieving the labels and convert them to dummy variables:
label_dummies = pd.get_dummies(df[LABELS])

# Creating training and test sets for the model:
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only, label_dummies, size = 0.2, seed = 123)

# Printing out the information about the sets:
print('X_train info: {}'.format(X_train.info()))
print('\nX_test_info: {}'.format(X_test.info()))
print('\ny_train info: {}'.format(y_train.info()))
print('\ny_test info: {}'.format(y_test.info()))

# Training a model:
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier 

# Instantiate the classifier
clf = OneVsRestClassifier(LogisticRegression())

# Fit the classifier to the training data:
clf.fit(X_train, y_train)

# Checking for accuracy:
print('Accuracy: {}'.format(clf.score(X_test, y_test)))

# Checking the holdout data:
holdout = pd.read_csv('Holdout.csv', index_col = 0)

# Generating predictions:
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))

# Checking the Logg Loss function to optimize the predictions
# The lower LogLoss the better model evaluation
prediction_df = pd.DataFrame(columns = pd.get_dummies(df[LABELS]).columns, 
							 index = holdout.index, 
							 data = predictions)

# Saving the predictions_df to csv
prediction_df.to_csv('predictions.csv')

# Submit the predictions for scoring:
score = score_submission(pred_path = 'predictions.csv')

# Printing out the score
# The result is 1.91 and the baseline 2.04
print('Your model, trained with numeric data only, yields logloss score: {}'.format(score))

# Another model using basic NLP vectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Token pattern:
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Filling missing values in df.Position_Extra
df.Position_Extra.fillna('', inplace = True)

# Instantiate the CountVectorizer
vec_alphanumeric = CountVectorizer(token_pattern = TOKENS_ALPHANUMERIC)

# Fit to the data
vec_alphanumeric.fit(df.Position_Extra)

# Print the number of tokens and first 15 tokens
msg = 'There are {} tokens in Position_Extra if we split on non-alpha numeric.'
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names()[:15])

def combine_text_columns(data_frame, to_drop = NUMERIC_COLUMNS + LABELS):
	"""Converts all text in each row of data_frame to single vector"""

	# Drop non-text columns that are in the df
	to_drop = set(to_drop) & set(data_frame.columns.tolist())
	text_data = data_frame.drop(to_drop, axis = 1)

	# Replace nans with blank spaces
	text_data.fillna("", inplace = True)

	# Join all text items in a row that have a space in between
	return text_data.apply(lamda x: " ".join(x), axis = 1)


# Basic token patter
TOKENS_BASIC = '\\S+(?=\\s+)'

# basic CountVectorizer
vec_basic = CountVectorizer(token_pattern = TOKENS_BASIC)

# alphanumeric CountVectorizer
vec_alphanumeric = CountVectorizer(token_pattern = TOKENS_ALPHANUMERIC)

# Text Vector
text_vector = combine_text_columns(df)

# Fit and transform vec_basic
vec_basic.fit_transform(text_vector)

# Print number of tokens of vec_basic
print('There are {} tokens in the dataset.'.format(len(vec_basic.get_feature_names())))

# Fit and transform vec_alphanumeric
vec_alphanumeric.fit_transform(text_vector)

# Print number of tokens of vec_alphanumeric
print('There are {} alpha-numeric tokens in the dataset.'.format(len(vec_alphanumeric.get_feature_names())))

# Trying new model

# Complete the pipeline
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction import CountVectorizer
from sklearn.preprocessing import FunctionTransformer


pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)

from sklearn.ensemble import RandomForestClassifier

# Edit model step in pipeline
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier())
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
