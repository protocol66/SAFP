import matplotlib.pyplot as plt
 
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


def main():
    clf_search_space = ClfSearchSpace()
    clf_testers = [ClassifierTester(feature_extr_fn=FeatureExtractor.method1),
                   ClassifierTester(feature_extr_fn=FeatureExtractor.method2)]
    # add confusion matrix to metrics
    for tester in clf_testers:
        tester.metrics.append('confusion_matrix')
            
    # create classifiers
    clf = {
        'KN': KNeighborsClassifier(),
        # 'LDA': LDA(),
        # 'GNB': GaussianNB()
        'SVM': svm.SVC(),
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
    
    # sort scores
    for i in range(len(clf_testers)):
        ncv_scores[i] = {k: v for k, v in sorted(ncv_scores[i].items(), key=lambda item: item[1], reverse=True)} 
    
    # print best 2 classifiers for each feature extractor
    print("\n\nBest 2 classifiers for each feature extractor")
    for i in range(len(clf_testers)):
        top2 = list(ncv_scores[i])[:2]
        print(f"Feature Extractor {i}: {top2[0]} score {ncv_scores[i][top2[0]]}, {top2[1]} score {ncv_scores[i][top2[1]]}")
    
    print("\n\nRunning cross validation on best classifiers to find best hyperparameters")
    # use cross validate to obtain best classifiers for each feature extractor-clf combination
    best_clfs_metrics = []
    for i in range(len(clf_testers)):
        best_clfs_metrics.append({})
        for c in list(ncv_scores[i])[:2]:
            best_clf = clf_testers[i].cross_validate(clf[c], clf_search_space.get_search_space(c), scoring='f1_macro')
            best_clfs_metrics[i][c] = clf_testers[i].test_classifier(best_clf)
    
    print("\n\n Scores for best classifiers")        
    # plot confusion matrix for best classifiers
    for i in range(len(clf_testers)):
        for c in best_clfs_metrics[i]:
            print(f"Feature Extractor {i} - {c}: Accuracy {best_clfs_metrics[i][c]['accuracy']} F1 {best_clfs_metrics[i][c]['f1']} Precision {best_clfs_metrics[i][c]['precision']} Recall {best_clfs_metrics[i][c]['recall']}")
            cm_disp = ConfusionMatrixDisplay(confusion_matrix=best_clfs_metrics[i][c]['confusion_matrix']).plot()
            cm_disp.ax_.set_title(f"Feature Extractor {i} - {c}")
            cm_disp.figure_.set_size_inches(10, 10)
            plt.savefig(f"confusion_matrix_{i}_{c}.pdf")
    plt.show()
if __name__ == '__main__':
    main()