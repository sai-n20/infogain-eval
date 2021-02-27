import os
import numpy as np
from collections import Counter
import sys

__location__ = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))

fileInputList = list()
attributeList = list()
classList = list()
entropyList = list()
classDict = dict()
infoGain = 0
datasetEntropy = 0

def readFile():
	with open(os.path.join(__location__, 'golf.txt')) as dataFile:
		for line in dataFile:
			fileInputList.append(list())
			fileInputList[len(fileInputList) - 1] = line.replace('\n', '').replace(' ', '').lower().split(',')
	dataFile.close()


readFile()

for line in fileInputList:
    classList.append(line[len(line) - 1])
    line.pop(len(line) - 1)

for attr in fileInputList[0]:
    attributeList.append(list())

for line in fileInputList:
    for index, attr in enumerate(line):
        attributeList[index].append(attr)

classCounter = Counter(classList)

for className in np.unique(classList):
    classDict[className] = 0


for className in np.unique(classList):
    datasetEntropy += classCounter[className]/len(fileInputList) * np.log2(classCounter[className]/len(fileInputList))
datasetEntropy = -datasetEntropy
print(datasetEntropy)


for column in attributeList:
    columnEntropyValue = 0
    for attr in np.unique(column):
        attrEntropyValue = 0
        indexes = [i for i, e in enumerate(column) if e == attr]
        for i in indexes:
            classDict[classList[i]] += 1
        for className in np.unique(classList):
            if(classDict[className] > 0):
                attrEntropyValue += classDict[className]/len(indexes) * np.log2(classDict[className]/len(indexes))
        columnEntropyValue += len(indexes)/len(fileInputList) * -attrEntropyValue
        for className in np.unique(classList):
            classDict[className] = 0
    entropyList.append(columnEntropyValue)

print(entropyList)

if(len(sys.argv) > 1 and int(sys.argv[1]) < len(entropyList)):
    infoGain = datasetEntropy - entropyList[int(sys.argv[1])]
    with open('result.txt', 'w') as output:
        output.write("(IG {})".format(infoGain))
    output.close()