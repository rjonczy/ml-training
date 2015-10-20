'''
Created on Oct 19, 2015
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: rjonczy
'''


from numpy import *
import operator


def createDataSet():
 	group = array( [ [1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1] ] )
 	labels = ['A', 'A', 'B', 'B']
 	return group, labels

'''
For every point in our dataset:
	calculate the distance between inX and the current point
	sort the distances in increasing order
	take k items with lowest distances to inX
	find the majority class among these items
	return the majority class as our prediction for the class of inX
'''
def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()

	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

	return sortedClassCount[0][0]

def file2matrix(filename):
	dictionary = { 'largeDoses':3, 'smallDoses':2, 'didntLike':1 }
	fr = open(filename)
	#get number of lines
	numberOfLines = len(fr.readlines())

	#create NumPy matrix to return
	returnMat = zeros((numberOfLines, 3))
	classLabelVector = []
	fr = open(filename)

	index = 0 
	for line in fr.readlines():
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		if(listFromLine[-1].isdigit()):
			classLabelVector.append(int(listFromLine[-1]))
		else:
			#print dictionary.get(listFromLine[-1])			
			classLabelVector.append(dictionary.get(listFromLine[-1]))
			# print len(classLabelVector)
		index += 1

	return returnMat, classLabelVector 

