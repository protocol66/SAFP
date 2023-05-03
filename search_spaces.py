from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class ClfSearchSpace():
    # Shotgun approach, one of these should work
    dt_space = {'clf__criterion': ['gini', 'entropy'],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 5],
                'clf__class_weight': [None, 'balanced'],
                'clf__max_depth': [None, 5, 10, 20, 50, 100]}
    rf_space = {'clf__n_estimators': [10, 50, 100],
                'clf__criterion': ['gini', 'entropy'],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 5],
                'clf__class_weight': ['balanced', 'balanced_subsample', None],
                'clf__bootstrap': [True, False]}
    ada_space = {'clf__estimator': [DecisionTreeClassifier(),],
                                    # RandomForestClassifier(n_estimators=10, criterion='gini',
                                                            # class_weight='balanced', bootstrap=True),],
                'clf__n_estimators': [10, 50, 100, 200],
                'clf__learning_rate': [0.1, 0.5, 0.9, 1.0]}
    svm_space = {'clf__C': [0.1, 0.5, 1, 5],
                 'clf__kernel': ['rbf', 'linear'],
                 'clf__gamma': ['scale', 'auto', .01, 0.001, 0.0001],
                 'clf__class_weight': [None, 'balanced']}
    kn_space = {'clf__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
                'clf__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}
    mlp_space = {'clf__activation': ['tanh', 'relu'],
                 'clf__solver': ['sgd', 'adam'],
                 'clf__alpha': [0.0001, 0.001, 0.01],
                 'clf__hidden_layer_sizes': [(10,), (50,), (100,), (200,), (50, 50), (100, 100)],
                 'clf__learning_rate': ['constant', 'invscaling', 'adaptive']}
    lda_space = [{'clf__solver': ['svd']},
                 {'clf__solver': ['lsqr', 'eigen'],
                  'clf__shrinkage': ['auto', None]}]

    search_space = {'DT': dt_space,
                    'RF': rf_space,
                    'ET': rf_space,
                    'Ada': ada_space,
                    'SVM': svm_space,
                    'KN': kn_space,
                    'MLP': mlp_space,
                    'LDA': lda_space,
                    }
    
    def get_search_space(self, clf_name):
        return self.search_space[clf_name]