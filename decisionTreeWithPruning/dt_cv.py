# version 1.0
from typing import List
import dt_core
import dt_global
from dt_core import *


def cv_pre_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for pre-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """  
    trainAccu = [0] * len(value_list)
    validAccu = [0] * len(value_list)
    for i in range(len(folds)):
        trainingSet = []
        for k in range(len(folds)):
            if k != i:
                trainingSet += folds[k]
        validationSet = folds[i]
        tree = dt_core.learn_dt(trainingSet,dt_global.feature_names[:-1].copy())
        for l in range(len(value_list)):
            depth = value_list[l]
            accuracyTraining = get_prediction_accuracy(tree,trainingSet, depth)
            accuracyValid = get_prediction_accuracy(tree, validationSet, depth)
            trainAccu[l] += accuracyTraining
            validAccu[l] += accuracyValid
            
    for i in range(len(trainAccu)):
        trainAccu[i] = trainAccu[i]/len(folds)
    for i in range(len(validAccu)):
        validAccu[i] = validAccu[i]/len(folds)
    return trainAccu, validAccu


def cv_post_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for post-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """ 

    trainAccu = [0] * len(value_list)
    validAccu = [0] * len(value_list)
    for i in range(len(folds)):
        trainingSet = []
        for k in range(len(folds)):
            if k != i:
                trainingSet += folds[k]
        validationSet = folds[i]
        tree = dt_core.learn_dt(trainingSet,dt_global.feature_names[:-1].copy())
        for l in range(len(value_list)):
            minGain = value_list[l]
            post_prune(tree,minGain)

            accuracyTraining = get_prediction_accuracy(tree,trainingSet)
            
            accuracyValid = get_prediction_accuracy(tree, validationSet)
            trainAccu[l] += accuracyTraining
            validAccu[l] += accuracyValid
    for i in range(len(trainAccu)):
        trainAccu[i] = trainAccu[i]/len(folds)
    for i in range(len(validAccu)):
        validAccu[i] = validAccu[i]/len(folds)



    return trainAccu, validAccu
