import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


from dataloader import Dataset
from feature_extractor import FeatureExtractor


# Functions for finding Accuracy of each classifier
class ClassifierTester():
	def __init__(self, feature_extr_fn=None):
		if feature_extr_fn is None:
			self.feature_extr_fn = lambda x: x
		else:
			self.feature_extr_fn = feature_extr_fn
     
		X_train, Y_train, _ = Dataset(path='Project2data', split='train')[:]
		X_test, Y_test, _ = Dataset(path='Project2data', split='test')[:]
  
		self.X_train = []
		self.X_test = []
  
		for i in range(len(X_train)):
			self.X_train.append(self.feature_extr_fn(X_train[i]))
		
		for i in range(len(X_test)):
			self.X_test.append(self.feature_extr_fn(X_test[i]))
   
		self.X_train = np.array(self.X_train)
		self.X_test = np.array(self.X_test)
		self.Y_train = np.array(Y_train)
		self.Y_test = np.array(Y_test)
  
		self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, Y_train, train_size=0.7, shuffle=True, random_state=42)
  
	

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
		
tester = ClassifierTester(feature_extr_fn=FeatureExtractor.method1)
print('Finished processing data')
# knn_acc = tester.KNN_test(1)
# print(f'KNN Accuracy: {knn_acc}')
tester.AccuracyAssessment()
