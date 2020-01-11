#!/usr/bin/env python3
import sys
import queue
from collections import defaultdict
from decisionnode import *
from printtree import *

def read(file_name):
    data = open(file_name)
    dataset = []
    for line in data.readlines():
        words = line.split('\n')[0].split('\t')
        dataset.append(words)
    return dataset

def unique_counts(part):
    """
    # Create counts of possible results
    # (the last column of each row is
    # the result)
    """
    results = defaultdict(int)
    for row in part:
        results[row[-1]] += 1
    return dict(results)


def gini_impurity(part):
    total = len(part)
    results = unique_counts(part)
    imp = 1
    for v in results.values():
        imp -= (v / float(total)) **2
    return imp


def entropy(part):
    from math import log
    total = len(part)
    results = unique_counts(part)
    ent = 0
    for v in results.values():
        p = v / float(total)
        ent -= p * log(p, 2)
    return ent

def divideset(part, column, value):
    set1=[]
    set2=[]
    if isinstance(value, int):
        split_function = lambda prot : int(prot[column]) >= value
    elif isinstance(value, float):
        split_function = lambda prot : float(prot[column]) >= value
    else:
        split_function = lambda prot : prot[column] == value
    for v in part:
        if split_function(v):
            set1.append(v)
        else:
            set2.append(v)
    return (set1,set2)

def build_tree(part, scoref=entropy, beta=0):
    if(len(part) == 0): return decisionnode()
    best_gain = 0
    best_criteria = None 
    best_sets = None
    
    columns = len(part[0]) -1 
    for elem in part:
        for i in range(columns):
            try:
                (set1, set2) = divideset(part, i, float(elem[i]))
            except ValueError:
                (set1, set2) = divideset(part, i, elem[i])
            total = len(part)
            pr = len(set1)/ float(total)
            pl = len(set2)/float(total)

            gain = scoref(part) - pr * scoref(set1) - pl * scoref(set2)
            if(gain > best_gain):
                best_gain = gain
                best_criteria = (i, elem[i])
                best_sets = (set1, set2)

    if(best_gain > beta):
        tree_r = build_tree(best_sets[0], scoref, beta)
        tree_l = build_tree(best_sets[1], scoref, beta)
        return decisionnode(best_criteria[0], best_criteria[1],tb=tree_r, fb=tree_l, gain=best_gain)
    else:
        return decisionnode(results=unique_counts(part), gain=best_gain)


def buildtree_iter(part, scoref=entropy, beta=0):
    if len(part)==0: return decisionnode()
    node_list = []
    sets_list = [[part, None, None]]
    while sets_list:
        best_criteria = None
        best_sets = None
        best_gain = 0
        current_node = sets_list.pop(0)
        data_set = current_node[0]
        father = current_node[1]
        side = current_node[2]
        update_best_gain = False  
        if scoref(data_set) > beta:
            for row in range(len(data_set)):
                for column in range(len(data_set[row])-1):
                    try:
                        set1,set2=divideset(data_set,column,float(data_set[row][column]))
                    except ValueError:
                        set1,set2=divideset(data_set,column,data_set[row][column]) 
                    total = len(data_set)
                    pr = len(set1)/ float(total)
                    pl = len(set2)/float(total)
                    current_gain = scoref(data_set) - pr * scoref(set1) - pl * scoref(set2)
                    if best_gain < current_gain:
                        best_gain = current_gain
                        best_sets = (set1,set2)
                        best_criteria = (column, data_set[row][column])
                        update_best_gain=True
            if update_best_gain:
                node = decisionnode(col=best_criteria[0], value=best_criteria[1])
                node_list.append([node, father, side])
                sets_list.append([best_sets[0], node, True])
                sets_list.append([best_sets[1], node ,False])
            else:
                node = decisionnode(results=unique_counts(data_set))
                node_list.append([node,father,side])
        else:
            node = decisionnode(results=unique_counts(data_set))
            node_list.append([node,father,side])
    
    for node1 in node_list:
        for node2 in node_list:
            if node1[0]==node2[1]:
                if node2[2]==True:
                    node1[0].tb=node2[0]
                else:
                    node1[0].fb=node2[0]
    
    return node_list[0][0]

def classify(obj, tree):
    dataset = read(sys.argv[1])
    dataset.append(obj)
    nodes = []
    nodes.append([tree,dataset])
    while nodes:
        current_node=nodes.pop(0)
        set1,set2=divideset(current_node[1], current_node[0].col, current_node[0].value)
        if current_node[0].results is not None and obj in current_node[1]:
            leaf_node=current_node[1].remove(obj)
            return unique_counts([data for data in current_node[1] if data != obj])
        else:
            if current_node[0].fb is not None:
                nodes.append([current_node[0].fb,set2])
            if current_node[0].tb is not None:
                nodes.append([current_node[0].tb,set1])


def test_performance(traning_set, test_set):
    pass


def array_equal(array1,array2):
    array1.sort()
    array2.sort()

    if len(array1) != len(array2):
        return False

    for i in range(len(array1)):
        if array1[i] != array2[i]:
            return False
    
    return True

def split_set(dat_file, percentage):
    split_index=((len(dat_file))*percentage)//100
    return dat_file[:split_index],dat_file[(split_index):]
    
def getLeafNodes(dat_file, tree):
    if tree.results != None:
        return [dat_file]
    else:
        set1,set2 = divideset(dat_file, tree.col, tree.value)
        return getLeafNodes(set1, tree.tb)+getLeafNodes(set2, tree.fb)

def prune(tree, threshold):
    '''print("-----")
    printtree(tree)
    print("Col:" + str(tree.col))
    print("Val:" + str(tree.value))
    print("Res:" + str(tree.results))
    print("Fb:" + str(tree.tb))
    print("Tb:" + str(tree.fb))
    print("Gain:" + str(tree.gain))'''
    if tree.tb == None: return False 
    if tree.tb.results != None and tree.fb.results != None:
        if(tree.gain < threshold):
            tree.col = (tree.tb.col, tree.fb.col)
            tree.value = None
            tree.results = merge_dicts(tree.tb.results,
                                        tree.fb.results)
            tree.tb = None
            tree.fb = None
            tree.gain = -1 
            print("He prunejat") 
            return True
        return False
    else:
        if(tree.tb.results == None): prunedT = prune(tree.tb, threshold)
        else: prunedT = False
        if(tree.fb.results == None): prunedF = prune(tree.fb, threshold)
        else: prunedF = False
        return prunedT or prunedF

def merge_dicts(dict1, dict2):
    for key in dict1:
        if key in dict2:
            dict2[key] += dict1[key]
        else:
            dict2[key] = dict1[key]
    return dict2

def main_2(): #Main ian
    data = read(sys.argv[1])
    tree = build_tree(part=data, beta=0)
    printtree(tree)
    print(" ------------- ")
    threshold = 0.82
    pruned = True
    while pruned:
        pruned = prune(tree, threshold)
        printtree(tree)
        print(pruned)
        print("-----")
def main_1(): #Main que tenies (quim), borrar despres
    dat_file = read(sys.argv[1])
    #counts = unique_counts(dat_file)
    #gini = gini_impurity(dat_file)
    #ent = entropy(dat_file)
    tree = build_tree(part=dat_file, beta=0)
    printtree(tree)
    #classification = classify(['facebook','New Zealand','no','22','None'], tree)
    #print(classification)
    tree_iter = buildtree_iter(part=dat_file, beta=0)
    printtree(tree_iter)
    #training_set_percentage = 98
    #training_set, test_set = split_set(dat_file, training_set_percentage)
    #correctly_classified=test_performance(training_set, test_set)
    #print(correctly_classified)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 treepredict.py data_file") 
        sys.exit()
    main_1()
    #main_2()
    


