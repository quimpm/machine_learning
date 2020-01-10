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

#TODO: Borrar buildtree2 i arreglar l'iteratiu
'''def buildtree2(part, scoref=entropy, beta=0):
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
'''
def buildtree_iter(part, scoref=entropy, beta=0):
    if len(part)==0: return decisionnode()
    best_criteria = scoref(part)
    best_sets = ()
    best_elem = ()
    node_list = []
    sets_list = [[part, None, None]]
    while sets_list:
        current_node = sets_list.pop(0)
        data_set = current_node[0]
        father = current_node[1]
        side = current_node[2]
        best_criteria = scoref(data_set)
        update_best_criteria = False #En cas d'estar en un node fulla amb impuresa mínima més gran que beta
        if best_criteria > beta:
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
                        update_best_criteria=True
            if update_best_criteria:
                node = decisionnode(col=best_elem[1], value=data_set[best_elem[0]][best_elem[1]])
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
            return [data for data in current_node[1] if data != obj]
        else:
            if current_node[0].fb is not None:
                nodes.append([current_node[0].fb,set2])
            if current_node[0].tb is not None:
                nodes.append([current_node[0].tb,set1])

def test_performance(training_set, test_set):
    full_data_set = read(sys.argv[1])
    original_tree = build_tree(full_data_set)
    training_tree = build_tree(training_set)
    print("Original Tree---------------------------------------------------")
    printtree(original_tree)
    print("Training Tree---------------------------------------------------")
    printtree(training_tree)
    leafs_original = getLeafNodes(full_data_set, original_tree)
    leafs_training = getLeafNodes(training_set, original_tree)
    totalNodes=len(full_data_set)
    correctly_classified=0
    for data in test_set:
        leaf_set=classify(data, training_tree) #a dincs de classify agafa el dataset sencer i per aixó no fuciona
        for i in range(len(leafs_training)):
            if array_equal(leaf_set,leafs_training[i]):
                print('hotal')
                leafs_training[i].append(data)
    for i in range(len(leafs_training)):
        for node in leafs_training[i]:
            if node in leafs_original[i]:
                correctly_classified+=1
    return correctly_classified/totalNodes*100

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
    print(tree.col)
    print(tree.value)
    print(tree.results)
    print(tree.tb)
    print(tree.fb)
    print(tree.gain)
    print(tree.tb.gain)
    print(tree.tb.tb.gain)
    print(tree.tb.tb.tb.gain)
    print(tree.tb.tb.tb.results)
    return False

def main_1(): #Main que tenies (quim), borrar despres
    dat_file = read(sys.argv[1])
    #counts = unique_counts(dat_file)
    #gini = gini_impurity(dat_file)
    #ent = entropy(dat_file)
    tree = build_tree(part=dat_file, beta=0)
    #printtree(tree)
    #classification = classify(['facebook','New Zealand','no','22','None'], tree)
    #print(classification)
    #tree_iter = buildtree_iter(part=dat_file, beta=0)
    #printtree(tree_iter)
    training_set_percentage = 98
    training_set, test_set = split_set(dat_file, training_set_percentage)
    correctly_classified=test_performance(training_set, test_set)
    print(correctly_classified)

def main_2(): #Main ian
    data = read(sys.argv[1])
    tree = build_tree(part=data, beta=0)
    printtree(tree)
    prune(tree, 2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 treepredict.py data_file") 
        sys.exit()
    #main_1()
    main_2()
    
    


