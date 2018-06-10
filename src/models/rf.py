#!/usr/bin/env python

#############################################################
#
# rf.py
#
# Author : Miravet-Verde, Samuel
# Written : 02/17/2016
# Last updated : 05/11/2016
#
# Basic script including functions to run a random forest
# classifier analysis
#
#############################################################

#####################
#   PACKAGES LOAD   #
#####################

import sys, os
# import utils
import numpy as np
from numpy import genfromtxt, savetxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit, LeaveOneOut
from scipy import interp
import matplotlib.pyplot as plt

# To autofit the figure sizes
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


#####################
# GENERAL FUNCTIONS #
#####################

def feature_weights(rf, feat_dict, weightsxfold, add='', add_string=''):
    """
    Set of functions to explore the weights of the different
    features in a random forest classifier
    """

    # Measure importances
    importances = rf.feature_importances_

    fo = open('/Users/yann/repos/hackcancer/intelcancer/src/features/'+add+'weights.txt','a')
    fo.write("Feature ranking tree "+add_string+" :\n----------\n")
    inv_map = {v: k for k, v in feat_dict.iteritems()}
    to_write = []
    for i in range(0, len(importances)):
        to_write.append([inv_map[i], importances[i]])
    to_write.sort(key=lambda x:x[1], reverse=True) # Sort by second element
    c = 1
    for el in to_write:
        fo.write(str(c)+'.'+el[0]+' '+str(el[1])+'\n')
        c += 1

        if el[0] not in weightsxfold:
            weightsxfold[el[0]]=[el[1]]
        else:
            weightsxfold[el[0]].append(el[1])

    fo.write('\n')
    fo.close()

    return weightsxfold


def print_weights(weightsxfold, add=''):
    """
    Given the weights per fold, prints the varplot for it (including bar errors)
    """
    dic = {k:[np.mean(v), np.std(v)] for k, v in weightsxfold.iteritems()}

    feats = []
    means = []
    stdev = []
    for feat, stats in sorted(dic.items(), key=lambda i: i[1][0]):
        feats.append(feat)
        means.append(stats[0])
        stdev.append(stats[1])

    # Plot
    plt.barh(range(len(feats)), means, color='g', xerr=stdev, ecolor='g', align='center', alpha=0.5)
    plt.yticks(range(len(feats)), feats, fontsize=13)
    plt.ylim([-1, len(feats)])
    plt.xlabel('Variance explained', fontsize=13)
    plt.title('weights, folds:'+str(len(weightsxfold.values()[0])))
    plt.savefig('/Users/yann/repos/hackcancer/intelcancer/src/visualization/'+add+'weights_barplot_'+str(len(weightsxfold.values()[0]))+'.png')
    plt.close()


def random_forest(X, y, n_est, folds=0, test_size=0.2, add=''):
    """
    Main fucntion to run a random forest classification problem
    You only need to pass two arrays coming directly from
    the training set generation

    X = features
    y = labels
    test_size = portion of the data used to test

    feat_names is a list value used to remove those columns we don't need to include
    """

    # Remove features we don't want in the analysis
    # if to_exclude != None:
    #     X = utils.remove_column(X, to_exclude)

    # define the method and fit the classifier
    classifier = RandomForestClassifier(n_estimators=n_est, oob_score=1, n_jobs=-1, random_state=50, max_features="auto", min_samples_leaf=5)

    if folds == 0 or folds == None:
        classifier.fit(X, y)
    else:
        # Run the classifier including cross-validation and plot
        # ROC curves

        # Using kfold
        # cv = StratifiedKFold(y, n_folds=folds, shuffle=True, random_state=np.random.RandomState(0))

        cv = StratifiedShuffleSplit(y, n_iter=folds, test_size=test_size)

        # To compute the roc and auc:
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr  = []

        # To plot the weigths
        # try:
        #     os.system('rm /Users/yann/repos/hackcancer/intelcancer/src/features/'+add+'weights.txt')
        # except:
        #     pass

        weightsxfold = {}

        for i, (train, test) in enumerate(cv):
            probs = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # ROC and area under the curve
            fpr, tpr, thresholds = roc_curve(y[test], probs[:,1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            roc_auc   = auc(fpr, tpr)

            #Study the weight of each feature:
            add_string = '_'+str(i+1)
            # feature_weights(classifier, feat_dict, weightsxfold, add ,add_string)

            # Plot it
            plt.plot(fpr, tpr, lw=1)

        # plot the random hypothesis
        plt.plot([0,1], [0,1], '--', color=(0.6, 0.6, 0.6), label='Random')

        # Basic stats plus plot of the mean roc
        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        # Plot the roc of the classifier
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC, folds = '+str(folds))
        plt.legend(loc="lower right")
        nm = '/Users/yann/repos/hackcancer/intelcancer/src/visualization/'+add+'roc_'+str(folds)+'.png'
        plt.savefig(nm)
        plt.close()

        # Plot the weights:
        # print_weights(weightsxfold, add=add)


    return classifier

#####################
#      CLASSES      #
#####################

#####################
# TEST & CHECK AREA #
#####################

if '__main__' == __name__:
    pass
