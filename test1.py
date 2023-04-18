# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:27:07 2022

@author: Ann
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, f_classif
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from numpy import array 

from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFpr, chi2,SelectFdr,SelectFpr,SelectFwe


X, y = load_iris ( return_X_y= True)

E = np.random.RandomState(42).\
    uniform(0, 0.1, size = (X.shape[0], 20))
    
X = np.hstack((X,E))

X_train, X_test,y_train, y_test = train_test_split(
    X,y, stratify = y, random_state = 0)

selector = SelectKBest(f_classif)
selector.fit (X_train, y_train)
scores = -np.log10 (selector.pvalues_)#lon->nho
scores /=scores.max()#scale lai


import matplotlib.pyplot as plt

X_indices = np.arange (X.shape[-1])
plt.figure(1)
plt.clf()
plt.bar (X_indices - 0.05, scores, 
         width = 0.2)
plt.title ('featuer univariate score')
plt.xlabel ('feature number')
plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
plt.show()

#Compare with SVM

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


clf = make_pipeline (MinMaxScaler(), 
                     LinearSVC())
clf.fit(X_train, y_train)
print(
    "Classification accuracy without selecting features: {:.3f}".format(
        clf.score(X_test, y_test)
    )
)
svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
svm_weights /= svm_weights.sum()

#-----------------------
#after univariate feature selection 
clf_selected = make_pipeline(SelectKBest(f_classif, k=4), MinMaxScaler(), LinearSVC())
clf_selected.fit(X_train, y_train)
print(
    "Classification accuracy after univariate feature selection: {:.3f}".format(
        clf_selected.score(X_test, y_test)
    )
)

svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
svm_weights_selected /= svm_weights_selected.sum()


#plot
#Classification accuracy after univariate feature selection: 0.868

plt.bar(
    X_indices - 0.45, scores, width=0.2, label=r"Univariate score ($-Log(p_{value})$)"
)

plt.bar(X_indices - 0.25, svm_weights, width=0.2, label="SVM weight")

plt.bar(
    np.arange (4) - 0.05,
    svm_weights_selected,
    width=0.2,
    label="SVM weights after selection",
)

plt.title("Comparing feature selection")
plt.xlabel("Feature number")
plt.yticks(())
plt.axis("tight")
plt.legend(loc="upper right")
plt.show()



# test!###########################################################


#Ranking of pixels with RFE

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target
#create RFE object and rank each pixel
svc = SVC(kernel = 'linear', C= 1)
rfe = RFE (estimator = svc, 
           n_features_to_select= 1, 
           step = 1)
rfe.fit(X,y)
ranking = rfe.ranking_.reshape(digits.images[1].shape)
len(rfe.support_)

X.shape
digits.images[1].shape

# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()


support_ = np.ones(n_features, dtype=np.bool)

np.ones(3, dtype = np.bool)


hasattr(SVC, 'coef_')

getattr(SVC, 'coef_')

a = getattr(rfe, 'coef_', None)

#support vector machine
at = svc.fit(X,y)

getattr(at, 'coef_')



# test!###########################################################
#with cross validation 




import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification


X, y = make_classification(
    n_samples=1000,
    n_features=25,
    n_informative=3,
    n_redundant=2,
    n_repeated=0,
    n_classes=8,
    n_clusters_per_class=1,
    random_state=0,
)

svc= SVC (kernel = 'linear', probability = True)
min_features_to_select = 1
rfecv = RFECV (
    estimator=svc,
    step =1,
    cv = StratifiedKFold(2),
    scoring = 'roc_auc_ovo',
    min_features_to_select= min_features_to_select)

rfecv.fit(X,y)

#Plot number of features vs. Cross-validation scores

plt.figure()
plt.xlabel ('Number of features selected')
plt.ylabel ('Cross validation score (accuary')
plt.plot (
    range (min_features_to_select, 
           len (rfecv.grid_scores_)+
           min_features_to_select),
    rfecv.grid_scores_,)
plt.show()

rfecv.grid_scores_

sklearn.metrics.SCORERS.keys()
from sklearn.metrics import SCORERS


#-------------------------------------------------------------------------------
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris ( return_X_y = True)
X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.4, random_state=0)

X_train.shape, y_train.shape

X_test.shape, y_test.shape


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)


from sklearn.model_selection import cross_val_score
clf = svm.SVC (kernel = 'linear', C = 1,
               random_state=42)
scores = cross_val_score(clf, X, y, cv = 5)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

##---------------------------
#make piple line 

from sklearn import preprocessing 
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, 
                     test_size = 0.4,
                     random_state = 0)
from sklearn.pipeline import make_pipeline
clf = make_pipeline( 
    preprocessing.StandardScaler(),
    svm.SVC (C = 1))

#The cross_validate function and multiple metric evaluation


#---------------------

from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

scoring = ['precision_macro', 
           'recall_macro', 'accuracy']

clf = svm.SVC (kernel = 'linear', 
               C = 1, 
               random_state = 0)
scores = cross_validate (clf, X, y,
                         scoring = scoring )

scores.keys()
scores['test_precision_macro']
set(y)

from sklearn.metrics import make_scorer
scoring = {'prec_macro':'precision_macro',
           'rec_macro': make_scorer(
               recall_score,
               average = 'macro')}

scores = cross_validate ( 
    clf, X, y, scoring = scoring, 
    cv = 5, return_train_score = True)
sorted (scores.keys())

scores = cross_validate (clf,X, y,
                scoring = 'precision_macro',
                cv = 5, 
                return_estimator = True)


#----------------------------
#roc with k fold cross validation 


X, y = X[y!=2], y[y!=2]

n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold


cv = StratifiedKFold(n_splits = 6)
classifier = svm.SVC(kernel = 'linear',
                     probability=True, 
                     random_state=random_state)

tprs = []
aucs = []

mean_fpr = np.linspace (0,1,100)

fig, ax = plt.subplots()
for i , (train, test) in enumerate (
        cv.split(X, y)): 
    classifier.fit (X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X[test],
        y[test], 
        name = 'ROC fold {}'.format (i),
        alpha = 0.3,
        lw = 1,
        ax = ax)
    interp_tpr = np.interp (
        mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
 
mean_tpr= np.mean (tprs, axis =0)
mean_tpr[-1] =1
mean_auc = auc ( mean_fpr, mean_tpr)
std_auc = np.std(aucs) 

ax.plot ( 
    mean_fpr,
    mean_tpr, 
    color = 'b', 
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc), 
    lw = 2, 
    alpha = 0.8,
    )   


std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between ( 
    mean_fpr, 
    tprs_lower, 
    tprs_upper, 
    color = 'grey',
    alpha = 0.2,
    label=r"$\pm$ 1 std. dev.")

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic example",
)
ax.legend(loc="lower right")
plt.show()


#----------------------------------------
from sklearn import datasets
import numpy as np
import pandas as pd

digits = datasets.load_digits()



n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target == 8


print(
    f"The number of images is {X.shape[0]} and each image contains {X.shape[1]} pixels"
)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


scores = ["precision", "recall"]  

import pandas as pd

def print_dataframe (filtered_cv_results): 
    for mean_precision, std_precision, mean_recall, std_recall, params in zip(
        filtered_cv_results["mean_test_precision"],
        filtered_cv_results["std_test_precision"],
        filtered_cv_results["mean_test_recall"],
        filtered_cv_results["std_test_recall"],
        filtered_cv_results["params"],
    ):
        print(
            f"precision: {mean_precision:0.3f} (±{std_precision:0.03f}),"
            f" recall: {mean_recall:0.3f} (±{std_recall:0.03f}),"
            f" for {params}"
        )
    print()

def refit_strategy(cv_results): 
    """Define the strategy to select the best estimator.

    The strategy defined here is to filter-out all results below a precision threshold
    of 0.98, rank the remaining by recall and keep all models with one standard
    deviation of the best by recall. Once these models are selected, we can select the
    fastest model to predict.

    Parameters
    ----------
    cv_results : dict of numpy (masked) ndarrays
        CV results as returned by the `GridSearchCV`.

    Returns
    -------
    best_index : int
        The index of the best estimator as it appears in `cv_results`.
    """
    precision_threshold = 0.98
    cv_results_  = pd.dataframe (cv_results)
    print ( 'All grid-search results:')
    
    high_precision_cv_results = cv_results_[]








  
    
    
    
    
    
    
    
    
    
    
    
