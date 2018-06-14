import numpy as np
import dill as pickle

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

EPS = 1E-8



def calc_WAcc(y, y_pred, flag_balanced=False, weight=0.5):
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
    if not flag_balanced:
        return (tp+tn) / (tn+fp+fn+tp+EPS)
    else:
        return weight * (tp/(tp+fn+EPS)) + (1-weight) * (tn/(tn+fp+EPS))

    
def calc_feature_importance(feature, label):
    ginis_feature = np.array([])
    for i in range(feature.shape[1]):
        dt = DecisionTreeClassifier(max_depth=1)
        dt.fit(feature[:, i].reshape((-1, 1)), label[:])
        
        threshold = dt.tree_.threshold[0]
        gini = (dt.tree_.impurity * dt.tree_.n_node_samples * [1, -1, -1]).sum()

        ginis_feature = np.append(ginis_feature, gini)
        
    return ginis_feature
    
    
def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))
    return

    
def load_model(filename):
    return pickle.load(open(filename, 'rb'))