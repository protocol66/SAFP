# Project 2 - 5/4/23
# Joshua Adams, Weston Beebe, Parth Patel, Jonathan Sanderson, Samuel Sylvester

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, RepeatedKFold, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
# from sklearn import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, make_scorer, precision_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support


from dataloader import Dataset
from feature_extractor import FeatureExtractor

metric_lut = {'accuracy': accuracy_score,
              'recall': recall_score,
              'precision': precision_score,
              'f1': f1_score,
              'roc_auc': roc_auc_score,
              'confusion_matrix': confusion_matrix,
              'precision_recall_fscore_support': precision_recall_fscore_support}

# Functions for finding Accuracy of each classifier
class ClassifierTester():
	metrics = ['accuracy', 'recall', 'precision', 'f1']  # metrics to use for scoring during testing
    
	def __init__(self, feature_extr_fn=None, base_pipeline=None):
		if feature_extr_fn is None:
			self.feature_extr_fn = lambda x: x
		else:
			self.feature_extr_fn = feature_extr_fn

		if base_pipeline is None:
			# create empty pipeline
			self.base_pipeline = Pipeline([('pass', FunctionTransformer(lambda x: x))])
		else:
			self.base_pipeline = base_pipeline
   
		# load data
		X_train, Y_train, _ = Dataset(path='Project2data', split='train')[:]
		X_test, Y_test, _ = Dataset(path='Project2data', split='test')[:]
  
		self.X_train = []
		self.X_test = []
  
		for i in range(len(X_train)):
			self.X_train.append(self.feature_extr_fn(X_train[i]))
		
		for i in range(len(X_test)):
			self.X_test.append(self.feature_extr_fn(X_test[i]))
   
		self.X_train_full = np.array(self.X_train)
		self.X_test = np.array(self.X_test)
		self.Y_train_full = np.array(Y_train)
		self.Y_test = np.array(Y_test)

		# TODO: why is Y_train not class variable?
		self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train_full, Y_train, train_size=0.7, shuffle=True, random_state=42)
	
	def _obtain_metrics(self, y_true, y_pred):
		metric_results = {}
		# determine if multiclass or binary
		if len(np.unique(y_true)) > 2:
			average = 'macro'
		else:
			average = 'binary'
   
		for m in self.metrics:
			# only pass average if it is a valid argument (is there a better way to do this?)
			if m in ['recall', 'precision', 'f1']: # TODO: move list outside of function?
				metric_results[m] = metric_lut[m](y_true, y_pred, average=average)
			elif m == 'roc_auc':
				metric_results[m] = metric_lut[m](y_true, y_pred, average=average, multi_class='ovo')
			else:
				metric_results[m] = metric_lut[m](y_true, y_pred)
		return metric_results
 
	
	def test_classifier(self, clf):
		# append classifier to pipeline
		pipeline = Pipeline([('base', self.base_pipeline), ('clf', clf)])
		# pipeline = Pipeline([('clf', clf)])
		pipeline.fit(self.X_train, self.Y_train)

		y_pred = pipeline.predict(self.X_test)
		metric_results = self._obtain_metrics(self.Y_test, y_pred)
  
		return metric_results

	def _inner_cv(self, clf, search_space, scoring='accuracy'):
		cv = KFold(n_splits=5, shuffle=True, random_state=42)
		search = GridSearchCV(clf, search_space, scoring=scoring, cv=cv, n_jobs=-1, refit=True, error_score=np.NaN)
		return search
 
	def cross_validate(self, clf, search_space, scoring='accuracy'):
		# create pipeline
		pipeline = Pipeline([('base', self.base_pipeline), ('clf', clf)])
		# pipeline = Pipeline([('clf', clf)])

		search = self._inner_cv(pipeline, search_space, scoring=scoring)
		search.fit(self.X_train_full, self.Y_train_full)
  
		print(f"Best parameters for {clf} with {scoring} {search.best_score_}:")
		print(search.best_params_)
		# return best estimator
		return search.best_estimator_
  
	def nested_cv_score(self, clf, search_space, inner_scoring='accuracy', outer_scoring='accuracy'):
		# nested cross validation
		cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
  
		# create pipeline
		pipeline = Pipeline([('base', self.base_pipeline), ('clf', clf)])

		# create grid search
		search = self._inner_cv(pipeline, search_space, scoring=inner_scoring)
		# perform nested cross validation
		scores = cross_validate(search, self.X_train_full, self.Y_train_full, scoring=outer_scoring, cv=cv_outer, n_jobs=-1, error_score=np.NaN)

		# return mean of scores
		for m in scores:
			scores[m] = np.mean(scores[m])
		return scores
