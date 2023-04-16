# version 1.0
import math
from typing import List
from anytree import Node
from collections import deque
import dt_global
from dt_provided import less_than, less_than_or_equal_to
import dt_provided



def get_splits(examples: List, feature: str) -> List[float]:
    """
    Given some examples and a feature, returns a list of potential split point values for the feature.
    
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :return: a list of potential split point values 
    :rtype: List[float]
    """ 
    
    res = []
    index = dt_global.feature_names.index(feature)
    examples.sort(key= lambda x: x[index])
    lastValue = examples[0][index]
    for example in examples:
        if (less_than (lastValue,example[index])):
            res.append((example[index] + lastValue)/2)
            lastValue = example[index]
        
    return res

def totalLabelCount(examples: List) -> dict:
    labels = {}
    for example in examples:
         currentLabel = example[-1]
         if (currentLabel in labels.keys()):
            labels[currentLabel] = labels[currentLabel] + 1
         else:
            labels[currentLabel] = 1
    return labels

def HCal(labels:dict, i: int, length: int) -> float:
    result = 0
    for label in labels.keys():
        result += labels[label]/i * math.log2(labels[label]/i)
    result *= i/length
    return -result
    
def calculateExpectInfoGain(examples: List, feature: str, labelCount: dict, length: int):
    hMin = math.inf
    splitPoint = None
    index = dt_global.feature_names.index(feature)
    examples.sort(key = lambda x: x[index])
    potentialSplitList = deque(get_splits(examples, feature))
    label = {}
    for i in range(length):
        if len(potentialSplitList) == 0:
            break
        currentExample = examples[i]
        if (less_than(potentialSplitList[0],currentExample[index])):
            curSplit = potentialSplitList.popleft()
            left =  HCal(label, i, length)
            #  set right dic
            rightDic = {}
            for key in labelCount.keys():
                count = labelCount[key]
                if (key in label.keys()):
                    count -= label[key]
                if count != 0:
                    rightDic[key] = count

            right = HCal(rightDic, length - i, length)
            h = left + right
            if (less_than(h,hMin)):
                hMin = h
                splitPoint = curSplit
        if currentExample[-1] in label.keys():
            label[currentExample[-1]] = label[currentExample[-1]] + 1
        else:
            label[currentExample[-1]] = 1

        
    return hMin, splitPoint
def choose_feature_split(examples: List, features: List[str]) -> (str, float, float):
    """
    Given some examples and some features,
    returns a feature and a split point value with the max expected information gain.

    If there are no valid split points for the remaining features, return None, -1, and -inf.

    Tie breaking rules:
    (1) With multiple split points, choose the one with the smallest value. 
    (2) With multiple features with the same info gain, choose the first feature in the list.

    :param examples: a set of examples
    :type examples: List[List[Any]]    
    :param features: a set of features
    :type features: List[str]
    :return: the best feature, the best split value, the max expected information gain
    :rtype: str, float, float
    """   
    totalCount = totalLabelCount(examples)
    length = len(examples)
    hbefore = HCal(totalCount,length, length)
    hafter =  math.inf
    splitValue = None
    bestfeature = None
    for feature in features:
        hMin, curSplit = calculateExpectInfoGain(examples, feature, totalCount, length)
        
        if (less_than(hMin,hafter)):
            splitValue = curSplit
            hafter = hMin
            bestfeature = feature
        
    if (bestfeature == None):
        return None, -1, -math.inf
    
    return bestfeature, splitValue, hbefore - hafter


def split_examples(examples: List, feature: str, split: float) -> (List, List):
    """
    Given some examples, a feature, and a split point,
    splits examples into two lists and return the two lists of examples.

    The first list of examples have their feature value <= split point.
    The second list of examples have their feature value > split point.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :param split: the split point
    :type split: float
    :return: two lists of examples split by the feature split
    :rtype: List[List[Any]], List[List[Any]]
    """ 
    index = dt_global.feature_names.index(feature)
    list1 = [x for x in examples if dt_provided.less_than_or_equal_to(x[index], split)]
    list2 = [x for x in examples if not dt_provided.less_than_or_equal_to(x[index], split)]
    return list1, list2

def split_node(cur_node: Node, examples: List, features: List[str], max_depth=math.inf):
    """
    Given a tree with cur_node as the root, some examples, some features, and the max depth,
    grows a tree to classify the examples using the features by using binary splits.

    If cur_node is at max_depth, makes cur_node a leaf node with majority decision and return.

    This function is recursive.

    :param cur_node: current node
    :type cur_node: Node
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the maximum depth of the tree
    :type max_depth: int
    """ 
    
    
    if (len(examples) == 0):
        cur_node.decision = cur_node.parent.decision
        return
    totalCount = totalLabelCount(examples)
    currentMax = -math.inf
    currentKey = math.inf
    for key, value in totalCount.items():
        if (value > currentMax):
            currentKey = key
            currentMax = value
        elif (value == currentMax):
            if (key < currentKey):
                currentKey = key
   
    cur_node.decision = currentKey
    if (len(totalCount.keys()) == 1 or cur_node.depth == max_depth):
        return    

    feature, splitValue, infoGain = choose_feature_split(examples, features)
    if (feature != None): 
        cur_node.feature = feature
        cur_node.split = splitValue
        cur_node.gain = infoGain
        lst1, lst2 = split_examples(examples, feature, splitValue)
        leftNode = Node(str(cur_node.depth + 1) + "-" + "1", cur_node)
        leftNode.parent = cur_node
        split_node(leftNode, lst1, features, max_depth)
        rightNode = Node(str(cur_node.depth + 1) + "-" + "2", cur_node)
        rightNode.parent = cur_node
        split_node(rightNode, lst2, features, max_depth)
                


def learn_dt(examples: List, features: List[str], max_depth=math.inf) -> Node:
    """
    Given some examples, some features, and the max depth,
    creates the root of a decision tree, and
    calls split_node to grow the tree to classify the examples using the features, and
    returns the root node.

    This function is a wrapper for split_node.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the max depth of the tree
    :type max_depth: int, default math.inf
    :return: the root of the tree
    :rtype: Node
    """ 
    node = Node("1")
    split_node(node, examples ,features, max_depth)
    return node


def predict(cur_node: Node, example, max_depth=math.inf) -> int:
    
    """
    Given a tree with cur_node as its root, an example, and optionally a max depth,
    returns a prediction for the example based on the tree.

    If max_depth is provided and we haven't reached a leaf node at the max depth, 
    return the majority decision at this node.
    
    This function is recursive.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param cur_node: cur_node of a decision tree
    :type cur_node: Node
    :param example: one example
    :type example: List[Any]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :return: the decision for the given example
    :rtype: int
    """ 
    if (cur_node.is_leaf or cur_node.depth == max_depth):
        
        return cur_node.decision
    
    feature = cur_node.feature
    splitValue = cur_node.split 
    index = dt_global.feature_names.index(feature)
    if (less_than_or_equal_to(example[index], splitValue)):
        return predict(cur_node.children[0], example, max_depth)
    else:
        return predict(cur_node.children[1], example, max_depth)
        

        
    


def get_prediction_accuracy(cur_node: Node, examples: List, max_depth=math.inf) -> float:
    """
    Given a tree with cur_node as the root, some examples, 
    and optionally the max depth,
    returns the accuracy by predicting the examples using the tree.

    The tree may be pruned by max_depth.

    :param cur_node: cur_node of the decision tree
    :type cur_node: Node
    :param examples: the set of examples. 
    :type examples: List[List[Any]]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :return: the prediction accuracy for the examples based on the cur_node
    :rtype: float
    """ 
    correct = 0
    for example in examples:
        prediction = predict(cur_node, example, max_depth)
        if (prediction == example[-1]):
            correct += 1
    return correct/len(examples)



def post_prune(cur_node: Node, min_info_gain: float):
    """
    Given a tree with cur_node as the root, and the minimum information gain,
    post prunes the tree using the minimum information gain criterion.

    This function is recursive.

    Let leaf parents denote all the nodes that only have leaf nodes as its descendants. 
    Go through all the leaf parents.
    If the information gain at a leaf parent is smaller than the pre-defined value,
    convert the leaf parent into a leaf node.
    Repeat until the information gain at every leaf parent is greater than
    or equal to the pre-defined value of the minimum information gain.

    :param cur_node: the current node
    :type cur_node: Node
    :param min_info_gain: the minimum information gain
    :type min_info_gain: float
    """
    if not cur_node.is_leaf :
        post_prune(cur_node.children[0],min_info_gain)
        post_prune(cur_node.children[1],min_info_gain)
        if cur_node.children[0].is_leaf and cur_node.children[1].is_leaf:
                if (less_than(cur_node.gain, min_info_gain)):
                    cur_node.children = []

        
             
        
            



