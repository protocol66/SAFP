import matplotlib.pyplot as plt
import pickle
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

from feature_extractor import FeatureExtractor
from Classifier_tester import ClassifierTester
from search_spaces import ClfSearchSpace


CLFS_SAVE_PATH = 'classifers.pkl'

def main():
    clf_search_space = ClfSearchSpace()
    clf_testers = [ClassifierTester(feature_extr_fn=FeatureExtractor.method1),
                   ClassifierTester(feature_extr_fn=FeatureExtractor.method2)]
            
    # create classifiers
    clf = {
        'KN': KNeighborsClassifier(),
        'GNB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(),
        'Ada': AdaBoostClassifier(),
    }
    ncv_scores = []
    for tester in clf_testers:
        ncv_scores.append({})
        tester_i = clf_testers.index(tester)
        print (f"Feature Extractor {tester_i}")
        for c in clf:
            if c == 'SVM' and tester_i == 1: # SVM is too slow/never completes for method2
                continue
            ncv_score = tester.nested_cv_score(clf[c], clf_search_space.get_search_space(c), outer_scoring='f1_macro', inner_scoring='f1_macro')
            print (c, ncv_score)
            ncv_scores[tester_i][c] = ncv_score['test_score'] 
    
    # average across feature extractors
    best_avg_ncv_scores = {}
    for c in clf:
        best_avg_ncv_scores[c] = sum([ncv_scores[i][c] for i in range(len(clf_testers))])/len(clf_testers)
    
    # sort average scores
    best_avg_ncv_scores = {k: v for k, v in sorted(best_avg_ncv_scores.items(), key=lambda item: item[1], reverse=True)}
    
    # find best 2 classifiers acrost all feature extractors
    # top2 = set()
    # for i in range(len(clf_testers)):
    #     top2.update(list(ncv_scores[i])[:2])
    
    # print best 2 classifiers for each feature extractor

    print("\n\nBest 2 classifiers for each feature extractor")
    top2 = list(best_avg_ncv_scores)[:2]
    for i in range(len(clf_testers)):
        print(f"Feature Extractor {i}: {top2[0]} score {ncv_scores[i][top2[0]]}, {top2[1]} score {ncv_scores[i][top2[1]]}")
    
    print("\n\nRunning cross validation on best classifiers to find best hyperparameters")
    # use cross validate to obtain best classifiers for each feature extractor-clf combination
    best_clfs_metrics = [{}]*len(clf_testers)
    save_clfs = [{}]*len(clf_testers)
    for i in range(len(clf_testers)):
        for c in list(top2)[:2]:
            best_clf = clf_testers[i].cross_validate(clf[c], clf_search_space.get_search_space(c), scoring='f1_macro')
            save_clfs[i][c] = best_clf['clf']   # get classifier from pipeline
            best_clfs_metrics[i][c] = clf_testers[i].test_classifier(best_clf)
    
    pickle.dump(save_clfs, open(CLFS_SAVE_PATH, 'wb'))
    print(f'Saved classifiers to {CLFS_SAVE_PATH}')
    
if __name__ == '__main__':
    main()