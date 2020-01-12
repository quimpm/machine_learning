#!/usr/bin/env python3
from math import sqrt
import dendrogram
import random
import sys


class bicluster:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id

def readfile(filename):
    with open(filename) as fd: 
        lines = [line for line in fd] 
    colnames = lines[0].strip().split('\t')[1:]
    rownames = []
    data = []
    for line in lines[1:]:
        p = line.strip().split('\t')
        #First column in each row is the rowname
        rownames.append(p[0])
        #The data for this row is the remainder
        data.append([float(x) for x in p[1:]])
    
    return rownames, colnames, data

def euclideansqrt(v1, v2):
    return sum(map(lambda x: (x[0]-x[1])**2, zip(v1,v2)))

def euclidean(v1, v2):
    return euclideansqrt(v1,v2)

def pearson(v1, v2):
    # Simple sums
    sum1 = sum(v1)
    sum2 = sum(v2)

    # Sums of the squares
    sum1Sq = sum([pow(v,2) for v in v1])
    sum2Sq = sum([pow(v,2) for v in v2])

    # Sums of the products
    pSum = sum([v1[i]*v2[i] for i in range(len(v1))])

    # Calculate r (Pearson score)
    num = pSum-(sum1*sum2/len(v1))
    den = sqrt((sum1Sq-pow(sum1,2)/len(v1)) * (sum2Sq-pow(sum2,2)/len(v1)))
    if den==0: return 0
    return 1.0-num/den

def hcluster(rows, distance=pearson):
    distances={} # cache of distance calculations
    currentclustid=-1 # non original clusters have negative id

    # Clusters are initially just the rows
    clust = [bicluster(rows[i], id=i) for i in range(len(rows))]
    while len(clust)>1: #Stop when there is only one cluster left
        lowestpair = (0,1)
        closest = distance(clust[0].vec, clust[1].vec)
        #loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                # distances is the cache of distance calculations
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
                
                d = distances[(clust[i].id, clust[j].id)]

                if d < closest:
                    closest = d
                    lowestpair = (i,j)
        
        # calculate the average of the two clusters
        mergevec = [ (clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i])/2.0
                for i in range(len(clust[0].vec))]
        # create new cluster
        newcluster = bicluster(mergevec, left=clust[lowestpair[0]], 
                right=clust[lowestpair[1]], distance = closest, id = currentclustid)
        currentclustid-=1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)

    return clust[0]

def printclust(clust, labels=None, n=0):
    # indent to make a hierarchy layout
    for _ in range(n): print(" ", end='')
    if clust.id < 0:
        # negative id means that this is branch
        print("-")
    else:
        # positive id means that this is endpoint
        if labels == None: print(clust.id, end='')
        else: print(labels[clust.id], end='')
        
    # now print the left and right branches
    if clust.left != None:
        printclust(clust.left, labels=labels, n=n+1)
    if clust.right != None:
        printclust(clust.right, labels=labels, n=n+1)

def rotatematrix(data):
    return [list(a) for a in list(zip(*data[::-1]))]

def dend(data, label = None, filename = None):
    clust = hcluster(data)
    dendrogram.drawdendrogram(clust, labels = label, jpeg="imageclust.jpg")

def kcluster(rows, distance = pearson, k=4):
    # Determine de min and max values for each point
    ranges = [(min([row[i] for row in rows]),max([row[i] for row in rows]))
            for i in range(len(rows[0]))]
    # Create k randomly placed centroids
    clusters = [[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0]
        for i in range(len(rows[0]))] for j in range(k)]
    lastmatches = None
    for _ in range(100):
        #print("Iteration " + str(t))
        bestmatches = [[] for _ in range(k)]
        # Find wich centroid is closest for each row
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row): bestmatch = i
            bestmatches[bestmatch].append(j)
        
        # If the results are the same as last time, done
        if bestmatches == lastmatches: break
        lastmatches = bestmatches
        
        # Move the centroids to de average of their members
        for i in range(k):
            avgs=[0.0]*len(rows[0])
            if(len(bestmatches[i])>0):
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]

                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs
    
    total_sum = calc_total_dist(clusters, bestmatches, rows, euclidean)
    return total_sum, bestmatches

def calc_total_dist(clusters, bestmatches, rows, distance = pearson):  
    total_sum = 0 
    for i in range(len(clusters)):
        for rowID in bestmatches[i]:
            d = distance(clusters[i], rows[rowID])
            total_sum += d**2
    return total_sum

def search_cluster(iterations):
    best_dist, best_clust = kcluster(data,k=10)
    for _ in range(iterations-1):
        total_dist, kclust = kcluster(data, k = 10)
        if total_dist < best_dist:
            best_dist = total_dist
            best_clust = kclust
    return best_dist, best_clust

if __name__=='__main__':
    blognames, words, data = readfile("blogdata.txt")
    if len(sys.argv) < 2:
        best_dist, best_clust = search_cluster(10)
    else:
        best_dist, best_clust = search_cluster(int(sys.argv[1]))
    print("Best distance: " + str(best_dist))
    print("Best clusters: ")
    print(best_clust)





