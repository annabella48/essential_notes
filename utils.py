# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 17:07:57 2022

@author: Ann
"""
#generate auc, roc_auc
from sklearn.metrics import roc_curve, auc

import numpy as np
import matplotlib.pyplot as plt


def roc_plot (y_actual1, y_pred1): 
    
    '''
    Plot roc when having y_predicted and y_actual and
        return fpr, tpr and roc_auc
    
    Input: -- y_actual: array
            -- y_pred: array
            
    output:--fpr: false positive rate (array)
            --tpr: true possive rate (array)
            --roc_auc: area under roc curve (float)
            
    '''
    
   

    fpr, tpr , _ = roc_curve(
            y_actual1,y_pred1)
    
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()
    
    return fpr, tpr, roc_auc

def plot_mean_roc_auc(X,y,n_splits = 6):
    ''' 
    Using SVM as classifier and CV of 
    StratfiedKFold and RocCurveDisplay packages
    Plot function for mean ROC and AUC with 
    input: X, y
    X, y must have train and test sets within its values
    n_splits by default is 6
    '''
    
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.metrics import auc
    from sklearn.metrics import RocCurveDisplay
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=n_splits)
    classifier = svm.SVC(kernel="linear", probability=True, random_state=random_state)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[test],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example",
    )
    ax.legend(loc="lower right")
    plt.show()

