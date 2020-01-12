#!/usr/bin/env python3
import sys
import queue
from collections import defaultdict
from decisionnode import *
from printtree import *
import random

def read(file_name):
    data = open(file_name)
    dataset = []
    for line in data.readlines():
        words = line.split('\n')[0].split(',')
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
    if isinstance(value, float):
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
        try:
            return decisionnode(best_criteria[0], float(best_criteria[1]),tb=tree_r, fb=tree_l, gain=best_gain)
        except ValueError:
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
        if best_gain > beta:
            try:
                node = decisionnode(col=best_criteria[0], value=float(best_criteria[1]), gain=best_gain)
            except ValueError:
                node = decisionnode(col=best_criteria[0], value=best_criteria[1], gain=best_gain)
            node_list.append([node, father, side])
            sets_list.append([best_sets[0], node, True])
            sets_list.append([best_sets[1], node ,False])
        else:
            node = decisionnode(results=unique_counts(data_set), gain=best_gain)
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
    if tree.results != None:
        return tree.results
    else:
        if isinstance(tree.value, float):
            if float(obj[tree.col]) >= tree.value:
                return classify(obj,tree.tb)
            else:
                return classify(obj,tree.fb)
        else:
            if obj[tree.col] == tree.value:
                return classify(obj,tree.tb)
            else:
                return classify(obj,tree.fb)
        

def test_performance(traning_set, test_set):
    tree = build_tree(traning_set)
    first_obj = test_set.pop(0)
    leaf_node=classify(first_obj, tree)
    percentage=calculate_percentage(leaf_node,first_obj)
    for obj in test_set:
        leaf_node=classify(obj, tree)
        percentage=(percentage+calculate_percentage(leaf_node, obj))/2
    return percentage

def calculate_percentage(leaf_node, obj):
    equals_to_objective = leaf_node.get(obj[-1]) if obj[-1] in leaf_node else 0
    total = 0
    for value in leaf_node.values():
        total += value
    return equals_to_objective/total * 100

def split_set(data_set, percentage):
    number_of_training=((len(data_set))*percentage)//100
    training_set=[]
    for i in range(number_of_training):
        index = random.randint(0,len(data_set)-1)
        training_set.append(data_set.pop(index))
    return training_set, data_set


def prune_tree(tree, threshold):
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
            return True
        return False
    else:
        pruneT = pruneF = True
        while(tree.tb.results == None and pruneT): pruneT = prune_tree(tree.tb, threshold)
        while(tree.fb.results == None and pruneF): pruneF = prune_tree(tree.fb, threshold)
        return pruneT and pruneF


def merge_dicts(dict1, dict2):
    for key in dict1:
        if key in dict2:
            dict2[key] += dict1[key]
        else:
            dict2[key] = dict1[key]
    return dict2

def main_2():
    dat_file = read(sys.argv[1])
    tree = build_tree(part=dat_file, beta=0)
    prune_tree(tree, 0.85)
    #prune(tree, 0.85)
    printtree(tree)

def main_1():
    dat_file = read(sys.argv[1])
    #tree = build_tree(part=dat_file, beta=0)
    #printtree(tree)
    #tree_iter = buildtree_iter(part=dat_file, beta=0)
    #printtree(tree_iter)
    traning_set, test_set=split_set(dat_file, 90)
    print(test_performance(traning_set, test_set))
    traning_set, test_set=split_set(dat_file, 70)
    print(test_performance(traning_set, test_set))
    #traning_set = read('training_set.data')
    #test_set = read('test_set.data')
    
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 treepredict.py data_file") 
        sys.exit()
    main_1()
    #main_2()
    


