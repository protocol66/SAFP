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
	metrics = ['accuracy', 'recall', 'precision', 'f1']
    
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
			else:
				metric_results[m] = metric_lut[m](y_true, y_pred)
		return metric_results
 
	
	def test_classifier(self, clf):
		# append classifier to pipeline
		pipeline = Pipeline([('base', self.base_pipeline), ('clf', clf)])
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

		scores = cross_validate(search, self.X_train_full, self.Y_train_full, scoring=outer_scoring, cv=cv_outer, n_jobs=-1, error_score=np.NaN)

		# return mean of scores
		for m in scores:
			scores[m] = np.mean(scores[m])
		return scores

	def KNN_test(self, k):
		
		knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
		knn.fit(self.X_train, self.Y_train)
		
		y_pred = knn.predict(self.X_val)
		accuracy = accuracy_score(self.Y_val, y_pred)

		return accuracy


	def DT_test(self):
		
		dt = DecisionTreeClassifier()
		dt = dt.fit(self.X_train, self.Y_train)
		
		y_pred = dt.predict(self.X_val)
		accuracy = accuracy_score(self.Y_val, y_pred)

		return accuracy

	# adaboost width decision tree
	def AdaBoost_test(self):
		
		dt = DecisionTreeClassifier()
		ab = AdaBoostClassifier(estimator=dt, n_estimators=50)
		ab = ab.fit(self.X_train, self.Y_train)
		
		y_pred = ab.predict(self.X_val)
		accuracy = accuracy_score(self.Y_val, y_pred)

		return accuracy

	def RF_test(self):
		
		rf = RandomForestClassifier(n_estimators=100)
		rf = rf.fit(self.X_train, self.Y_train)
		
		y_pred = rf.predict(self.X_val)
		accuracy = accuracy_score(self.Y_val, y_pred)

		return accuracy


	def SVM_test(self):
		
		sv = svm.SVC(kernel='linear', C=10)
		sv = sv.fit(self.X_train, self.Y_train)
		
		y_pred = sv.predict(self.X_val)
		accuracy = accuracy_score(self.Y_val, y_pred)

		return accuracy
		
	def NN_test(self, hidden_layers, max_it):

		NN = MLPClassifier(hidden_layer_sizes=hidden_layers, activation='tanh', solver='adam', max_iter=max_it)
		NN = NN.fit(self.X_train, self.Y_train)
		
		y_pred = NN.predict(self.X_val)
		accuracy = accuracy_score(self.Y_val, y_pred)

		return accuracy
		

	def NB_test(self):		
		
		NB = GaussianNB()
		NB = NB.fit(self.X_train, self.Y_train)
		
		y_pred = NB.predict(self.X_val)
		accuracy = accuracy_score(self.Y_val, y_pred)

		return accuracy


	def AccuracyAssessment(self):
		# # get dataset from other file
		# data = GET OUR DATA HERE
		
		k = 2
		hidden_layer_sizes = (1024,1024,1024)
		max_it = 100000
		
		
		print('KNN Accuracy with ', k, ' neighbors: ', self.KNN_test(k))
		print('DT Accuracy with depth of ', k, ': ', self.DT_test())
		print('AdaBoost Accuracy with depth of : ', self.AdaBoost_test())
		print('RF Accuracy: ', self.RF_test())
		print('SVM Accuracy: ', self.SVM_test())
		print('NN Accuracy with max iterations of ', max_it, 'and ', hidden_layer_sizes, 'hidden layers: ', self.NN_test(hidden_layer_sizes, max_it))
		print('NB Accuracy: ', self.NB_test())

if __name__ == '__main__':
	tester = ClassifierTester(feature_extr_fn=FeatureExtractor.method1)
	print('Finished processing data')
	# knn_acc = tester.KNN_test(1)
	# print(f'KNN Accuracy: {knn_acc}')
	tester.AccuracyAssessment()
