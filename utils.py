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



