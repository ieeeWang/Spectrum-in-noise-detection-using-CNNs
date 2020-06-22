# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:56:06 2020
sklearn.metrics.precision_recall_curve
@author: lwang
"""

import numpy as np
from sklearn.metrics import precision_recall_curve

#%%
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision

#%% Load Data and train model
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = fetch_openml(data_id=1464, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
clf.fit(X_train, y_train)


#%% Create ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

cm_display = ConfusionMatrixDisplay(cm, [0,1]).plot()

#%% Create ROC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay

y_score = clf.decision_function(X_test)
pos_label=clf.classes_[1]
fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pos_label)
AUC = auc(fpr, tpr)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=AUC, estimator_name='demo').plot()

#%% Create PR
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay

pos_label=clf.classes_[1]
prec, recall, _ = precision_recall_curve(y_test, y_score,
                                         pos_label=pos_label)
# alternative AUCpr (~AP), with a different computing method
AP = average_precision_score(y_test, y_score, pos_label=pos_label)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall, 
                                    average_precision=AP,
                                    estimator_name='demo').plot()
AUCpr = auc(x=recall, y=prec)

#%% Combining the display objects (ROC and PR) into a single plot
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

roc_display.plot(ax=ax1)
pr_display.plot(ax=ax2)
plt.show()


