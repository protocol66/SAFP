# Project 2 - 5/4/23
# Joshua Adams, Weston Beebe, Parth Patel, Jonathan Sanderson, Samuel Sylvester

import matplotlib.pyplot as plt
import pickle
import os

from sklearn.metrics import ConfusionMatrixDisplay

from feature_extractor import FeatureExtractor
from Classifier_tester import ClassifierTester
from train import CLFS_SAVE_PATH
    
RESULTS_DIR = 'results/'
    
def main():
    saved_clfs = pickle.load(open(CLFS_SAVE_PATH, 'rb'))
    
    # create testers with different feature extractors (same as in train.py)
    clf_testers = [ClassifierTester(feature_extr_fn=FeatureExtractor.method1),
                   ClassifierTester(feature_extr_fn=FeatureExtractor.method2)]
    
    # add confusion matrix to metrics in tester
    for tester in clf_testers:
        tester.metrics.append('confusion_matrix')
        # tester.metrics.append('roc_auc')
        
    # check if results directory exists, create one if needed
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    print("\n\nScores for best classifiers")        
    # plot confusion matrix for best classifiers
    for i in range(len(clf_testers)): # for each feature extractor
        for c in saved_clfs[i]:       # for each classifier
            metrics = clf_testers[i].test_classifier(saved_clfs[i][c])                           
            print(f"Feature Extractor {i} - {c:3s}: Accuracy {metrics['accuracy']:.3f} ",
                  f"F1 {metrics['f1']:.3f} ",
                  f"Precision {metrics['precision']:.3f} ",
                  f"Recall {metrics['recall']:.3f}")
            cm_disp = ConfusionMatrixDisplay(confusion_matrix=metrics['confusion_matrix']).plot()
            cm_disp.ax_.set_title(f"FE Method {i + 1} - {c}")
            cm_disp.figure_.set_size_inches(10, 10)
            cm_disp.figure_.tight_layout()
            plt.savefig(RESULTS_DIR + f"confusion_matrix_{i}_{c}.pdf")
            
        print('-' * 80)
    plt.show()
    
    
if __name__ == "__main__":
    main()