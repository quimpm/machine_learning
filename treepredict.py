#!/usr/bin/env python3
import sys
import queue
#import numpy as np
#import tree_class as tree
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
            
def buildtree(part, scoref=entropy, beta=0):
    if len(part)==0: return decisionnode()
    best_criteria = scoref(part)
    best_sets = ()
    best_elem = ()
    update_best_criteria = False #En cas d'estar en un node fulla amb impuresa mínima més gran que beta
    for row in range(len(part)):
        for column in range(len(part[row])-1):
            try:
                set1,set2=divideset(part,column,float(part[row][column]))
            except ValueError:
                set1,set2=divideset(part,column,part[row][column]) 
            current_criteria = max(scoref(set1),scoref(set2))
            if best_criteria > current_criteria:
                best_criteria = current_criteria
                best_sets = (set1,set2)
                best_elem = (row,column)
                update_best_criteria = True
    if best_criteria < beta:
        return decisionnode(results=unique_counts(part))
    else:
        if update_best_criteria == True:
            try:
                return decisionnode(col=best_elem[1], value=float(part[best_elem[0]][best_elem[1]]), tb=buildtree(best_sets[0], entropy, beta), fb=buildtree(best_sets[1], entropy, beta))
            except ValueError:
                return decisionnode(col=best_elem[1], value=part[best_elem[0]][best_elem[1]], tb=buildtree(best_sets[0], entropy, beta), fb=buildtree(best_sets[1], entropy, beta))
        else:
            return decisionnode(results=unique_counts(part))

def buildtree_iter(part, scoref=entropy, beta=0):
    if len(part)==0: return decisionnode()
    best_criteria = scoref(part)
    best_sets = ()
    best_elem = ()
    node_list = []
    sets_list = [part]
    while best_criteria > beta:
        data_set=sets_list.pop(0)
        print(best_criteria)
        for row in range(len(data_set)):
            for column in range(len(data_set[row])-1):
                try:
                    set1,set2=divideset(data_set,column,float(data_set[row][column]))
                except ValueError:
                    set1,set2=divideset(data_set,column,data_set[row][column]) 
                current_criteria = max(scoref(set1),scoref(set2))
                if best_criteria > current_criteria:
                    best_criteria = current_criteria
                    best_sets = (set1,set2)
                    best_elem = (row,column)
        print(best_sets)
        node_list.append([data_set[best_elem[0]][best_elem[1]], best_elem[1]])
        sets_list.append(best_sets[0])
        sets_list.append(best_sets[1])
    return node_list
        

def classify(obj, tree):
    dataset = read(sys.argv[1])
    dataset.append(obj)
    nodes = []
    nodes.append([tree,dataset])
    while nodes:
        current_node=nodes.pop(0)
        set1,set2=divideset(current_node[1], current_node[0].col, current_node[0].value)
        if current_node[0].results is not None and obj in current_node[1]:
            return current_node[1]
        else:
            if current_node[0].fb is not None:
                nodes.append([current_node[0].fb,set2])
            if current_node[0].tb is not None:
                nodes.append([current_node[0].tb,set1])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit()

    dat_file = read(sys.argv[1])
    #counts = unique_counts(dat_file)
    #gini = gini_impurity(dat_file)
    #ent = entropy(dat_file)
    #tree = buildtree(part=dat_file, beta=1)
    #printtree(tree)
    #classification=classify(['google','New Zealand','no','22','None'], tree)
    tree_iter=buildtree_iter(part=dat_file, beta=1.0)
    print(tree_iter)
    


